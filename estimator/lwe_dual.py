# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using dual attacks.

See :ref:`LWE Dual Attacks` for an introduction what is available.

"""

from functools import partial

from sage.all import oo, binomial, ceil, floor, sqrt, log, cached_function, RR, exp, pi, e, coth, tanh

from .reduction import delta as deltaf, cost as costf
from .simulator import normalize as simulator_normalize
from .util import local_minimum, early_abort_range, bisect_false_true, log_gh
from .cost import Cost
from .lwe_parameters import LWEParameters
from .prob import drop as prob_drop, amplify as prob_amplify
from .io import Logging
from .conf import red_cost_model as red_cost_model_default, mitm_opt as mitm_opt_default
from .errors import OutOfBoundsError, InsufficientSamplesError
from .nd import DiscreteGaussian, SparseTernary, sigmaf
from .lwe_guess import exhaustive_search, mitm, distinguish
from .conf import red_shape_model as red_shape_model_default
from .conf import red_simulator as red_simulator_default


class DualHybrid:
    """
    Estimate cost of solving LWE using dual attacks.
    """

    @staticmethod
    @cached_function
    def dual_reduce(
        delta: float,
        params: LWEParameters,
        zeta: int = 0,
        h1: int = 0,
        rho: float = 1.0,
        t: int = 0,
        log_level=None,
    ):
        """
        Produce new LWE sample using a dual vector on first `n-ζ` coordinates of the secret. The
        length of the dual vector is given by `δ` in root Hermite form and using a possible
        scaling factor, i.e. `|v| = ρ ⋅ δ^d * q^((n-ζ)/d)`.

        :param delta: Length of the vector in root Hermite form
        :param params: LWE parameters
        :param zeta: Dimension ζ ≥ 0 of new LWE instance
        :param h1: Number of non-zero components of the secret of the new LWE instance
        :param rho: Factor introduced by obtaining multiple dual vectors
        :returns: new ``LWEParameters`` and ``m``

        .. note :: This function assumes that the instance is normalized.

        """
        if not 0 <= zeta <= params.n:
            raise OutOfBoundsError(
                f"Splitting dimension {zeta} must be between 0 and n={params.n}."
            )

        # Compute new secret distribution
        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            if not 0 <= h1 <= h:
                raise OutOfBoundsError(f"Splitting weight {h1} must be between 0 and h={h}.")

            if type(params.Xs) is SparseTernary:
                # split the +1 and -1 entries in a balanced way.
                slv_Xs, red_Xs = params.Xs.split_balanced(zeta, h1)
            else:
                # TODO: Implement this for sparse secret that are not SparseTernary,
                # i.e. DiscreteGaussian with extremely small stddev.
                raise NotImplementedError(f"Unknown how to exploit sparsity of {params.Xs}")

            if h1 == h:
                # no reason to do lattice reduction if we assume
                # that the hw on the reduction part is 0
                return params.updated(Xs=slv_Xs, m=oo), 1
        else:
            # distribution is i.i.d. for each coordinate
            red_Xs = params.Xs.resize(params.n - zeta)
            slv_Xs = params.Xs.resize(zeta)

        c = red_Xs.stddev * params.q / params.Xe.stddev

        # see if we have optimally many samples (as in [INDOCRYPT:EspJouKha20]) available
        m_ = max(1, ceil(sqrt(red_Xs.n * log(c) / log(delta))) - red_Xs.n)
        m_ = min(params.m, m_)

        # apply the [AC:GuoJoh21] technique, m_ not optimal anymore?
        d = m_ + red_Xs.n
        rho /= 2 ** (t / d)

        # Compute new noise as in [INDOCRYPT:EspJouKha20]
        # ~ sigma_ = rho * red_Xs.stddev * delta ** (m_ + red_Xs.n) / c ** (m_ / (m_ + red_Xs.n))
        sigma_ = rho * red_Xs.stddev * delta**d / c ** (m_ / d)
        slv_Xe = DiscreteGaussian(params.q * sigma_)

        slv_params = LWEParameters(
            n=zeta,
            q=params.q,
            Xs=slv_Xs,
            Xe=slv_Xe,
        )

        # The m_ we compute there is the optimal number of samples that we pick from the input LWE
        # instance. We then need to return it because it determines the lattice dimension for the
        # reduction.

        return slv_params, m_

    @staticmethod
    @cached_function
    def cost(
        solver,
        params: LWEParameters,
        beta: int,
        zeta: int = 0,
        h1: int = 0,
        t: int = 0,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=None,
    ):
        """
        Computes the cost of the dual hybrid attack that dual reduces the LWE instance and then
        uses the given solver to solve the reduced instance.

        :param solver: Algorithm for solving the reduced instance
        :param params: LWE parameters
        :param beta: Block size used to produce short dual vectors for dual reduction
        :param zeta: Dimension ζ ≥ 0 of new LWE instance
        :param h1: Number of non-zero components of the secret of the new LWE instance
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction

        .. note :: This function assumes that the instance is normalized. It runs no optimization,
            it merely reports costs.

        """
        Logging.log("dual", log_level, f"β={beta}, ζ={zeta}, h1={h1}")

        delta = deltaf(beta)

        # only care about the scaling factor and don't know d yet -> use 2 * beta as dummy d
        rho = red_cost_model.short_vectors(beta=beta, d=2 * beta)[0]

        params_slv, m_ = DualHybrid.dual_reduce(
            delta, params, zeta, h1, rho, t, log_level=log_level + 1
        )
        Logging.log("dual", log_level + 1, f"red LWE instance: {repr(params_slv)}")

        if t:
            cost = DualHybrid.fft_solver(params_slv, success_probability, t)
        else:
            cost = solver(params_slv, success_probability)
        cost["beta"] = beta

        if oo in (cost['rop'], cost['m']):
            return cost

        d = m_ + params.n - zeta
        _, cost_red, N, sieve_dim = red_cost_model.short_vectors(beta, d, cost["m"])
        del cost["m"]
        Logging.log("dual", log_level + 2, f"red: {Cost(rop=cost_red)!r}")

        # Add the runtime cost of sieving in dimension `sieve_dim` possibly multiple times.
        cost["rop"] += cost_red

        # Add the memory cost of storing the `N` dual vectors, using `sieve_dim` many coefficients
        # (mod q) to represent them. Note that short dual vectors may actually be described by less
        # bits because its coefficients are generally small, so this is really an upper bound here.
        cost["mem"] += sieve_dim * N

        if d < params.n - zeta:
            raise RuntimeError(f"{d} < {params.n - zeta}, {params.n}, {zeta}, {m_}")

        Logging.log("dual", log_level, f"{repr(cost)}")

        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            probability = RR(prob_drop(params.n, h, zeta, h1))
            # don't need more samples to re-run attack, since we may
            # just guess different components of the secret
            cost = cost.repeat(prob_amplify(success_probability, probability))
        return cost + {'m': m_, 'd': d}

    @staticmethod
    def fft_solver(params, success_probability, t=0):
        """
        Estimate cost of solving LWE via the FFT distinguisher from [AC:GuoJoh21]_.

        :param params: LWE parameters
        :param success_probability: the targeted success probability
        :param t: the number of secret coordinates to guess mod 2.
            For t=0 this is similar to lwe_guess.ExhaustiveSearch.
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: memory requirement in integers mod q.
        - ``m``: Required number of samples to distinguish the correct solution with high probability.
        - ``t``: the number of secret coordinates to guess mod 2.

        .. note :: The parameter t only makes sense in the context of the dual attack,
            which is why this function is here and not in the lwe_guess module.
        """
        Cost.register_impermanent(t=False)

        # there are two stages: enumeration and distinguishing, so we split up the success_probability
        probability = sqrt(success_probability)

        try:
            size = params.Xs.support_size(probability)
            size_fft = 2**t
        except NotImplementedError:
            # not achieving required probability with search space
            # given our settings that means the search space is huge
            # so we approximate the cost with oo
            return Cost(rop=oo, mem=oo, m=1)

        sigma = params.Xe.stddev / params.q

        # Here, assume the Independence Heuristic, cf. [ia.cr/2023/302].
        # The minimal number of short dual vectors that is required to distinguish the correct
        # guess with probability at least `probability`:
        m_required = RR(
            4
            * exp(4 * pi * pi * sigma * sigma)
            * (log(size_fft * size) - log(log(1 / probability)))
        )

        if params.m < m_required:
            raise InsufficientSamplesError(
                f"Exhaustive search: Need {m_required} samples but only {params.m} available."
            )

        # Running a fast Walsh--Hadamard transform takes time proportional to t 2^t.
        runtime_cost = size * (t * size_fft)
        # Add the cost of updating the FFT tables for all of the enumeration targets.
        # Use "Efficient Updating of the FFT Input", [MATZOV, §5.4]:
        runtime_cost += size * (4 * m_required)

        # This is the number of entries the table should have. Note that it should support
        # (floating point) numbers in the range [-N, N], if ``N`` is the number of dual vectors.
        # However 32-bit floats are good enough in practice.
        memory_cost = size_fft

        return Cost(rop=runtime_cost, mem=memory_cost, m=m_required, t=t)

    @staticmethod
    def optimize_blocksize(
        solver,
        params: LWEParameters,
        zeta: int = 0,
        h1: int = 0,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=5,
        opt_step=8,
        fft=False,
    ):
        """
        Optimizes the cost of the dual hybrid attack over the block size β.

        :param solver: Algorithm for solving the reduced instance
        :param params: LWE parameters
        :param zeta: Dimension ζ ≥ 0 of new LWE instance
        :param h1: Number of non-zero components of the secret of the new LWE instance
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction
        :param opt_step: control robustness of optimizer
        :param fft: use the FFT distinguisher from [AC:GuoJoh21]_

        .. note :: This function assumes that the instance is normalized. ζ and h1 are fixed.

        """

        f_t = partial(
            DualHybrid.cost,
            solver=solver,
            params=params,
            zeta=zeta,
            h1=h1,
            success_probability=success_probability,
            red_cost_model=red_cost_model,
            log_level=log_level,
        )

        if fft is True:

            def f(beta):
                with local_minimum(0, params.n - zeta) as it:
                    for t in it:
                        it.update(f_t(beta=beta, t=t))
                    return it.y

        else:
            f = f_t

        # don't have a reliable upper bound for beta
        # we choose n - k arbitrarily and adjust later if
        # necessary
        beta_upper = min(max(params.n - zeta, 40), 1024)
        beta = beta_upper
        while beta == beta_upper:
            beta_upper *= 2
            with local_minimum(40, beta_upper, opt_step) as it:
                for beta in it:
                    it.update(f(beta=beta))
                for beta in it.neighborhood:
                    it.update(f(beta=beta))
                cost = it.y
            beta = cost["beta"]

        cost["zeta"] = zeta
        if params.Xs.is_sparse:
            cost["h_"] = h1
        return cost

    def __call__(
        self,
        solver,
        params: LWEParameters,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        opt_step=8,
        log_level=1,
        fft=False,
    ):
        """
        Optimizes the cost of the dual hybrid attack (using the given solver) over
        all attack parameters: block size β, splitting dimension ζ, and
        splitting weight h1 (in case the secret distribution is sparse). Since
        the cost function for the dual hybrid might only be convex in an approximate
        sense, the parameter ``opt_step`` allows to make the optimization procedure more
        robust against local irregularities (higher value) at the cost of a longer
        running time. In a nutshell, if the cost of the dual hybrid seems suspiciously
        high, try a larger ``opt_step`` (e.g. 4 or 8).

        :param solver: Algorithm for solving the reduced instance
        :param params: LWE parameters
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction
        :param opt_step: control robustness of optimizer
        :param fft: use the FFT distinguisher from [AC:GuoJoh21]_. (ignored for sparse secrets)

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: Total amount of memory used by solver (in elements mod q).
        - ``red``: Number of word operations in lattice reduction.
        - ``β``: BKZ block size.
        - ``ζ``: Number of guessed coordinates.
        - ``h1``: Number of non-zero components among guessed coordinates (if secret distribution is sparse)
        - ``prob``: Probability of success in guessing.
        - ``repetitions``: How often we are required to repeat the attack.
        - ``d``: Lattice dimension.
        - ``t``: Number of secrets to guess mod 2 (only if ``fft`` is ``True``)

        - When ζ = 1 this function essentially estimates the dual attack.
        - When ζ > 1 and ``solver`` is ``exhaustive_search`` this function estimates
            the hybrid attack as given in [INDOCRYPT:EspJouKha20]_
        - When ζ > 1 and ``solver`` is ``mitm`` this function estimates the dual MITM
            hybrid attack roughly following [IEEE:CHHS19]_

        EXAMPLES::

            >>> from estimator import *
            >>> from estimator.lwe_dual import dual_hybrid
            >>> params = LWE.Parameters(n=1024, q = 2**32, Xs=ND.Binary, Xe=ND.DiscreteGaussian(3.0))
            >>> LWE.dual(params)
            rop: ≈2^107.0, mem: ≈2^66.4, β: 264, m: 970, d: 1994, tag: dual
            >>> dual_hybrid(params)
            rop: ≈2^103.2, mem: ≈2^97.4, β: 250, m: 937, d: 1919, ζ: 42, tag: dual_hybrid
            >>> dual_hybrid(params, mitm_optimization=True)
            rop: ≈2^130.1, mem: ≈2^127.0, k: 120, β: 347, m: 1144, d: 2024, ζ: 144, tag: dual_mitm_hybrid
            >>> dual_hybrid(params, mitm_optimization="numerical")
            rop: ≈2^129.0, k: 1, mem: ≈2^131.0, β: 346, m: 1145, d: 2044, ζ: 125, tag: dual_mitm_hybrid

            >>> params = params.updated(Xs=ND.SparseTernary(64))
            >>> LWE.dual(params)
            rop: ≈2^103.4, mem: ≈2^63.9, β: 251, m: 904, d: 1928, tag: dual
            >>> dual_hybrid(params)
            rop: ≈2^91.6, mem: ≈2^77.2, β: 168, ↻: ≈2^11.2, m: 711, d: 1456, ζ: 279, h': 8, tag: dual_hybrid
            >>> dual_hybrid(params, mitm_optimization=True)
            rop: ≈2^98.7, mem: ≈2^78.6, k: 288, ↻: ≈2^19.6, β: 184, m: 737, d: 1284, ζ: 477, h': 17, tag: dual_mitm_...

            >>> params = params.updated(Xs=ND.CenteredBinomial(8))
            >>> LWE.dual(params)
            rop: ≈2^114.5, mem: ≈2^71.8, β: 291, m: 1103, d: 2127, tag: dual
            >>> dual_hybrid(params)
            rop: ≈2^113.6, mem: ≈2^103.5, β: 288, m: 1096, d: 2110, ζ: 10, tag: dual_hybrid
            >>> dual_hybrid(params, mitm_optimization=True)
            rop: ≈2^155.5, mem: ≈2^146.2, k: 34, β: 438, m: 1414, d: 2404, ζ: 34, tag: dual_mitm_hybrid

            >>> params = params.updated(Xs=ND.DiscreteGaussian(3.0))
            >>> LWE.dual(params)
            rop: ≈2^116.6, mem: ≈2^73.2, β: 299, m: 1142, d: 2166, tag: dual
            >>> dual_hybrid(params)
            rop: ≈2^116.2, mem: ≈2^105.8, β: 297, m: 1137, d: 2154, ζ: 7, tag: dual_hybrid
            >>> dual_hybrid(params, mitm_optimization=True)
            rop: ≈2^160.7, mem: ≈2^156.8, k: 25, β: 456, m: 1473, d: 2472, ζ: 25, tag: dual_mitm_hybrid

            >>> dual_hybrid(schemes.NTRUHPS2048509Enc)
            rop: ≈2^136.2, mem: ≈2^127.8, β: 356, ↻: 35, m: 434, d: 902, ζ: 40, h': 19, tag: dual_hybrid

            >>> LWE.dual(schemes.CHHS_4096_67)
            rop: ≈2^206.8, mem: ≈2^137.5, β: 616, m: ≈2^11.8, d: 7779, tag: dual

            >>> dual_hybrid(schemes.Kyber512, red_cost_model=RC.GJ21, fft=True)
            rop: ≈2^149.8, mem: ≈2^92.1, t: 75, β: 400, m: 510, d: 1000, ζ: 22, tag: dual_hybrid

        """
        Logging.log("dual", log_level, f"costing LWE instance: {repr(params)}")

        params = params.normalize()

        if params.Xs.is_sparse:
            def _optimize_blocksize(
                solver,
                params: LWEParameters,
                zeta: int = 0,
                success_probability: float = 0.99,
                red_cost_model=red_cost_model_default,
                log_level=None,
                fft=False,
            ):
                h = params.Xs.hamming_weight
                h1_min = max(0, h - (params.n - zeta))
                h1_max = min(zeta, h)
                if h1_min == h1_max:
                    h1_max = h1_min + 1
                Logging.log("dual", log_level, f"h1 ∈ [{h1_min},{h1_max}] (zeta={zeta})")
                with local_minimum(h1_min, h1_max, log_level=log_level + 1) as it:
                    for h1 in it:
                        # ignoring fft on purpose for sparse secrets
                        cost = self.optimize_blocksize(
                            h1=h1,
                            solver=solver,
                            params=params,
                            zeta=zeta,
                            success_probability=success_probability,
                            red_cost_model=red_cost_model,
                            log_level=log_level + 2,
                        )
                        it.update(cost)
                    return it.y

        else:
            _optimize_blocksize = self.optimize_blocksize

        f = partial(
            _optimize_blocksize,
            solver=solver,
            params=params,
            success_probability=success_probability,
            red_cost_model=red_cost_model,
            log_level=log_level + 1,
            fft=fft,
        )

        with local_minimum(1, params.n - 1, opt_step) as it:
            for zeta in it:
                it.update(f(zeta=zeta))
            for zeta in it.neighborhood:
                it.update(f(zeta=zeta))
            cost = it.y

        cost["problem"] = params
        return cost.sanity_check()


DH = DualHybrid()


class MATZOV:
    """
    See [AC:GuoJoh21]_ and [MATZOV22]_.
    """

    C_mul = 32**2  # p.37
    C_add = 5 * 32  # guessing based on C_mul

    @classmethod
    def T_fftf(cls, k, p):
        """
        The time complexity of the FFT in dimension `k` with modulus `p`.

        :param k: Dimension
        :param p: Modulus ≥ 2

        """
        return cls.C_mul * k * p ** (k + 1)  # Theorem 7.6, p.38

    @classmethod
    def T_tablef(cls, D):
        """
        Time complexity of updating the table in each iteration.

        :param D: Number of nonzero entries

        """
        return 4 * cls.C_add * D  # Theorem 7.6, p.39

    @classmethod
    def Nf(cls, params, m, beta_bkz, beta_sieve, k_enum, k_fft, p):
        """
        Required number of samples to distinguish with advantage.

        :param params: LWE parameters
        :param m:
        :param beta_bkz: Block size used for BKZ reduction
        :param beta_sieve: Block size used for sampling
        :param k_enum: Guessing dimension
        :param k_fft: FFT dimension
        :param p: FFT modulus

        """
        mu = 0.5
        k_lat = params.n - k_fft - k_enum  # p.15

        # p.39
        lsigma_s = (
            params.Xe.stddev ** (m / (m + k_lat))
            * (params.Xs.stddev * params.q) ** (k_lat / (m + k_lat))
            * sqrt(4 / 3.0)
            * sqrt(beta_sieve / 2 / pi / e)
            * deltaf(beta_bkz) ** (m + k_lat - beta_sieve)
        )

        # p.29, we're ignoring O()
        N = (
            exp(4 * (lsigma_s * pi / params.q) ** 2)
            * exp(k_fft / 3.0 * (params.Xs.stddev * pi / p) ** 2)
            * (k_enum * cls.Hf(params.Xs) + k_fft * log(p) + log(1 / mu))
        )

        return RR(N)

    @staticmethod
    def Hf(Xs):
        return RR(
            1 / 2 + log(sqrt(2 * pi) * Xs.stddev) + log(coth(pi**2 * Xs.stddev**2))
        ) / log(2.0)

    @classmethod
    def cost(
        cls,
        beta,
        params,
        m=None,
        p=2,
        k_enum=0,
        k_fft=0,
        beta_sieve=None,
        red_cost_model=red_cost_model_default,
    ):
        """
        Theorem 7.6

        """
        Cost.register_impermanent(beta_=False, t=False, N=False)

        if m is None:
            m = params.n

        k_lat = params.n - k_fft - k_enum  # p.15

        # We assume here that β_sieve ≈ β
        N = cls.Nf(
            params,
            m,
            beta,
            beta_sieve if beta_sieve else beta,
            k_enum,
            k_fft,
            p,
        )

        rho, T_sample, _, beta_sieve = red_cost_model.short_vectors(
            beta, N=N, d=k_lat + m, sieve_dim=beta_sieve
        )

        H = cls.Hf(params.Xs)

        coeff = 1 / (1 - exp(-1 / 2 / params.Xs.stddev**2))
        tmp_alpha = pi**2 * params.Xs.stddev**2
        tmp_a = exp(8 * tmp_alpha * exp(-2 * tmp_alpha) * tanh(tmp_alpha)).n(30)
        T_guess = coeff * (
            ((2 * tmp_a / sqrt(e)) ** k_enum)
            * (2 ** (k_enum * H))
            * (cls.T_fftf(k_fft, p) + cls.T_tablef(N))
        )

        cost = Cost(
            rop=T_sample + T_guess, red=T_sample, guess=T_guess,
            beta=beta, p=p, zeta=k_enum, t=k_fft, beta_=beta_sieve, N=N, m=m,
            problem=params,
        )

        return cost

    def __call__(
        self,
        params: LWEParameters,
        red_cost_model=red_cost_model_default,
        log_level=1,
    ):
        """
        Optimizes cost of dual attack as presented in [MATZOV22]_.

        See also [AC:GuoJoh21]_.

        :param params: LWE parameters
        :param red_cost_model: How to cost lattice reduction

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``red``: Number of word operations in lattice reduction and
                   short vector sampling.
        - ``guess``: Number of word operations in guessing and FFT.
        - ``β``: BKZ block size.
        - ``ζ``: Number of guessed coordinates.
        - ``t``: Number of coordinates in FFT part mod `p`.
        - ``d``: Lattice dimension.

        """
        params = params.normalize()

        for p in early_abort_range(2, params.q):
            for k_enum in early_abort_range(0, params.n, 10):
                for k_fft in early_abort_range(0, params.n - k_enum[0], 10):
                    # RC.ADPS16(1754, 1754) ~ 2^(512)
                    with local_minimum(40, min(params.n, 1754), log_level=log_level + 4) as it:
                        for beta in it:
                            cost = self.cost(
                                beta,
                                params,
                                p=p[0],
                                k_enum=k_enum[0],
                                k_fft=k_fft[0],
                                red_cost_model=red_cost_model,
                            )
                            it.update(cost)
                        Logging.log(
                            "dual",
                            log_level + 3,
                            f"t: {k_fft[0]}, {repr(it.y)}",
                        )
                        k_fft[1].update(it.y)
                Logging.log("dual", log_level + 2, f"ζ: {k_enum[0]}, {repr(k_fft[1].y)}")
                k_enum[1].update(k_fft[1].y)
            Logging.log("dual", log_level + 1, f"p:{p[0]}, {repr(k_enum[1].y)}")
            p[1].update(k_enum[1].y)
            # if t == 0 then p is irrelevant, so we early abort that loop if that's the case once we hit t==0 twice.
            if p[1].y["t"] == 0 and p[0] > 2:
                break
        Logging.log("dual", log_level, f"{repr(p[1].y)}")
        return p[1].y


matzov = MATZOV()


def dual(
    params: LWEParameters,
    success_probability: float = 0.99,
    red_cost_model=red_cost_model_default,
):
    """
    Dual attack as in [PQCBook:MicReg09]_.

    :param params: LWE parameters.
    :param success_probability: The success probability to target.
    :param red_cost_model: How to cost lattice reduction.

    The returned cost dictionary has the following entries:

    - ``rop``: Total number of word operations (≈ CPU cycles).
    - ``mem``: Total amount of memory used by solver (in elements mod q).
    - ``red``: Number of word operations in lattice reduction.
    - ``δ``: Root-Hermite factor targeted by lattice reduction.
    - ``β``: BKZ block size.
    - ``prob``: Probability of success in guessing.
    - ``repetitions``: How often we are required to repeat the attack.
    - ``d``: Lattice dimension.

    """
    ret = DH.optimize_blocksize(
        solver=distinguish,
        params=params,
        zeta=0,
        h1=0,
        success_probability=success_probability,
        red_cost_model=red_cost_model,
        log_level=1,
    )
    del ret["zeta"]
    if "h_" in ret:
        del ret["h_"]
    ret["tag"] = "dual"
    return ret


def dual_hybrid(
    params: LWEParameters,
    success_probability: float = 0.99,
    red_cost_model=red_cost_model_default,
    mitm_optimization=False,
    opt_step=8,
    fft=False,
):
    """
    Dual hybrid attack from [INDOCRYPT:EspJouKha20]_.

    :param params: LWE parameters.
    :param success_probability: The success probability to target.
    :param red_cost_model: How to cost lattice reduction.
    :param mitm_optimization: One of "analytical" or "numerical". If ``True`` a default from the
           ``conf`` module is picked, ``False`` disables MITM.
    :param opt_step: Control robustness of optimizer.
    :param fft: use the FFT distinguisher from [AC:GuoJoh21]_. (ignored for sparse secrets)

    The returned cost dictionary has the following entries:

    - ``rop``: Total number of word operations (≈ CPU cycles).
    - ``mem``: Total amount of memory used by solver (in elements mod q).
    - ``red``: Number of word operations in lattice reduction.
    - ``δ``: Root-Hermite factor targeted by lattice reduction.
    - ``β``: BKZ block size.
    - ``ζ``: Number of guessed coordinates.
    - ``h1``: Number of non-zero components among guessed coordinates (if secret distribution is sparse)
    - ``prob``: Probability of success in guessing.
    - ``repetitions``: How often we are required to repeat the attack.
    - ``d``: Lattice dimension.
    - ``t``: Number of secrets to guess mod 2 (only if ``fft`` is ``True``)
    """

    if mitm_optimization is True:
        mitm_optimization = mitm_opt_default

    if mitm_optimization:
        solver = partial(mitm, optimization=mitm_optimization)
    else:
        solver = exhaustive_search

    ret = DH(
        solver=solver,
        params=params,
        success_probability=success_probability,
        red_cost_model=red_cost_model,
        opt_step=opt_step,
        fft=fft,
    )
    if mitm_optimization:
        ret["tag"] = "dual_mitm_hybrid"
    else:
        ret["tag"] = "dual_hybrid"
    return ret


####################################################################################################
class CHHS19:
    """
    Estimate cost of solving LWE using the dual attack from [IEEE:CHHS19]_.
    """

    MEMORY_BOUND = 2**80

    @staticmethod
    @cached_function
    def cost_guessing(n: int, zeta: int, h: int, h0: int):
        """
        Return all data related to guessing `zeta` coefficients out of `n`, having a hamming weight of `h0` out of
        total `h`.

        :return: A tuple

        The returned tuple jonsists of the following four values:

        - time to search for all these keys (using MitM).
        - memory required to run this MitM algorithm.
        - probability that the secret splits in this way, assuming you apply a random permutation on the secret
          coefficients.
        - log(search space size)
        """
        # Split search space Odlyzko style:
        zeta_1, h_1 = (zeta + 1) // 2, (h0 + 1) // 2  # Take ceiling
        S_1 = binomial(zeta_1, h_1) * 2**h_1

        #
        # If you want to bound the memory cost, uncomment below:
        #
        # while S_1 > CHHS19.MEMORY_BOUND:
        #     h_1 -= 1
        #     zeta_1 = int(round(RR(zeta * h_1 / h)))
        #     S_1 = binomial(zeta_1, h_1) * 2**h_1

        zeta_2, h_2 = zeta - zeta_1, h0 - h_1
        S_2 = binomial(zeta_2, h_2) * 2**h_2

        time_search = S_1 + S_2
        mem_search = min(S_1, S_2)
        p_split = RR(binomial(zeta_1, h_1) * binomial(zeta_2, h_2) * binomial(n - zeta, h - h0) / binomial(n, h))
        log_search_space = RR(log(S_1) + log(S_2))
        return (time_search, mem_search, p_split, log_search_space)

    def cost(
        self,
        params: LWEParameters,
        beta: int,
        zeta: int,
        h0: int = None,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=1,
        only_works=False,
    ):
        """
        Cost of the [IEEE:CHHS19]_ attack.
        Estimate cost of solving LWE using the dual attack from [IEEE:CHHS19]_.
        :param params: LWE parameters
        :param beta: Block size used to produce short dual vectors
        :param zeta: Guessing dimension ζ ≥ 0.
        :param h0: Number of non-zero components expected in the guessed secret.
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction
        :param only_works: whether to return only if attack can work or not, instead of returning cost.

        .. note :: This is the lowest level function that runs no optimization, it merely reports
           costs.

        """
        n, h = params.n, params.Xs.hamming_weight
        delta = deltaf(beta)

        if h0 is None:
            # Pick h0 most balanced possible.
            h0 = int(round(RR(h * zeta / n)))

        # Compute complexity of this search space:
        t_search, mem_search, p_search, log_ss = self.cost_guessing(n, zeta, h, h0)

        if h0 == h:
            if only_works:
                return True
            # This basically boils down to MitM search
            cost = Cost(
                rop=params.m * t_search, mem=mem_search,
                beta=beta, zeta=zeta, h_=h0, prob=RR(p_search),
            )
            return cost.repeat(prob_amplify(success_probability, cost["prob"]))

        # xi is the scaling factor for the secret such that `xi s_2` and `e` have the same standard deviation,
        # where s_1 is the non-guessed secret, assumed to be of weight `h - h0`, and dimension `n - zeta`.
        var_s1 = (h - h0) / (n - zeta)
        xi = params.Xe.stddev / sqrt(var_s1)
        assert xi > 1
        m = min(params.m, int(round(RR(sqrt((n - zeta) * log(params.q / xi) / log(delta)) - (n - zeta)))))
        d = m + n - zeta

        log_det = (n - zeta) * log(params.q / xi)
        # ell: length of shortest vector after BKZ-beta
        ell = delta**d * exp(log_det / d)

        # <y2, xi s1> + <y1, e>,
        # where:
        # - (y1, y2) \in Lambda^\perp(A_2),
        # - `xi s1 \in xi ZZ^n` (stddev ~params.Xe.stddev),
        # - `e \in ZZ^m (stddev params.Xe.stddev).
        # B: infinity-norm bound on error e' of "new" LWE instance.
        B = 2.0 * sigmaf(params.Xe.stddev) * ell

        works = ell < params.q / 2.0 and B < params.q / 4.0
        if only_works:
            return works
        if not works:
            return Cost(rop=oo)

        # Number of dual vectors that we want (i.e. number of samples for "new" LWE instance)
        tau = int(round(RR(log_ss / (1.0 - 4 * B / params.q))))

        # Time for running BKZ with blocksize beta in dimension d.
        _, t_BKZ, tau_, beta_ = red_cost_model.short_vectors(beta, d, N=tau, B=log(params.q, 2))
        if tau_ != tau or beta_ != 2:
            # Just do it ourselves...
            t_BKZ = red_cost_model(beta, d, B=log(params.q, 2))
            t_BKZ += (tau - 1) * red_cost_model.LLL(d, log(params.q, 2))
            t_BKZ = RR(t_BKZ)
        # assert tau_ == tau and beta_ == 2
        # Assume we used LLL for the other vectors. Is not required for the algorithm.

        # Note: the paper [CHHS19]_ incorrectly reports that this probability should have exponent `m` in Theorem 1,
        # but it should be `tau`.
        p_error_bounded = (1.0 - 2 * exp(-4*pi))**tau

        cost = Cost(
            rop=t_BKZ + m * t_search, red=t_BKZ, mem=mem_search,
            beta=beta, zeta=zeta, h_=h0, d=d,
            prob=RR(p_search * p_error_bounded),
        )

        # print(f"beta={beta}, zeta={zeta}, h0={h0}: T_BKZ = {float(t_BKZ):.2e}, T_search = {float(time_search):.2e}")
        cost = cost.repeat(prob_amplify(success_probability, cost["prob"]))
        return cost + {'m': m, 'm_': tau}

    def minimal_zeta_needed(
        self,
        params: LWEParameters,
        beta: int,
        h0: int
    ):
        """
        Returns a pair (a, b) that indicates that the smallest `zeta` for which the dual attack works, satisfies:

            `a <= zeta <= b`.

        Note here that this bound on zeta is to make sure that lattice reduction finds short enough dual vectors in the
        remaining small-secret embedding lattice of dimension `m + n - zeta`, where `m` is chosen optimally, i.e.

            `m = sqrt((n - zeta) log(q/xi) / log(deltaf(beta))) - (n - zeta)`.
        """
        h = params.Xs.hamming_weight
        required_ell = params.q / (8 * sqrt(pi) * params.Xe.stddev)

        # n - zeta:
        RHS = 0.25 * log(required_ell)**2 / log(deltaf(beta))
        A = log(params.q * sqrt(h - h0) / params.Xe.stddev)

        x_0 = RR(RHS / A)
        # Perform one iteration of the Newton--Raphson root finding algorithm regarding the equation:
        #  f(x) = 2Ax - x log(x) - RHS = 0
        #  f'(x) = 2A - log(x) - 1
        #  x_{i+1} = x_i - f(x_i) / f'(x_i).
        x_1 = RR((2 * RHS - x_0) / (2 * A - log(x_0) - 1))

        # `n - zeta \in [x_0, x_1]` which corresponds to `\zeta \in [n - x_1, n - x_0]`.
        zeta_1 = max(0, min(params.n, int(ceil(RR(params.n - x_1)))))
        zeta_2 = max(0, min(params.n, int(ceil(RR(params.n - x_0)))))
        return [zeta_1, zeta_2]

    def cost_beta(
        self,
        params: LWEParameters,
        beta: int,
        h0: int = None,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=1,
    ):
        """
        Cost of the [IEEE:CHHS19]_ attack for a particular beta.
        Estimate cost of solving LWE using the dual attack from [IEEE:CHHS19]_.
        :param params: LWE parameters
        :param beta: Block size used to produce short dual vectors
        :param h0: Number of non-zero components expected in the guessed secret.
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction

        """
        # Optimize guessing dimension zeta.
        zeta_lo, max_zeta = self.minimal_zeta_needed(params, beta, h0)

        f = partial(
            self.cost,
            params=params,
            beta=beta,
            h0=h0,
            success_probability=success_probability,
            red_cost_model=red_cost_model,
            log_level=log_level + 1,
            only_works=True,
        )
        zeta_hi = max_zeta
        # Perform a binary search to find smallest zeta such that the attack can work.
        while zeta_hi - zeta_lo >= 2:
            zeta = (zeta_lo + zeta_hi) // 2
            if f(zeta=zeta):
                # The attack works.
                zeta_hi = zeta
            else:
                # The attack does not work.
                zeta_lo = zeta
        min_zeta = zeta_hi  # For zeta_hi, the attack works.
        Logging.log("dual", log_level, f"Range ζ is [{min_zeta}..{max_zeta}]")
        with local_minimum(min_zeta, max_zeta + 1, precision=4, log_level=log_level + 1) as it:
            f = partial(
                self.cost,
                params=params,
                beta=beta,
                success_probability=success_probability,
                red_cost_model=red_cost_model,
                log_level=log_level + 1,
            )

            # Cheat: use minimal zeta.
            return f(zeta=min_zeta)
            for zeta in it:
                it.update(f(zeta=zeta))
            # for zeta in it.neighborhood:
            #     it.update(f(zeta=zeta))
            cost = it.y
        return cost

    def cost_h0(
        self,
        params: LWEParameters,
        h0: int = None,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=1,
    ):
        """
        Cost of the [IEEE:CHHS19]_ attack for a particular h0.

        Estimate cost of solving LWE using the dual attack from [IEEE:CHHS19]_.
        :param params: LWE parameters
        :param h0: Number of non-zero components expected in the guessed secret.
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction

        """
        # Step 0. Establish a baseline when guessing half of the coefficients.
        max_beta = max(2 * params.n, 40)
        zeta_0 = params.n // 2

        # TODO: this part does not really depend on beta.

        with local_minimum(40, max_beta + 1, precision=5, log_level=log_level + 1) as it:
            f = partial(
                self.cost,
                params=params,
                zeta=zeta_0,
                h0=h0,
                success_probability=success_probability,
                red_cost_model=red_cost_model,
                log_level=log_level + 1,
            )

            for beta in it:
                it.update(f(beta=beta))
            max_beta = it.x
            cost = it.y

        Logging.log("dual", log_level, f"h0={h0} --> β_max = {max_beta}, cost = {repr(cost)}")

        # Step 1. Optimize blocksize beta (while also optimizing zeta).
        with local_minimum(40, max_beta + 1, precision=2, log_level=log_level + 1) as it:
            f = partial(
                self.cost_beta,
                params=params,
                h0=h0,
                success_probability=success_probability,
                red_cost_model=red_cost_model,
                log_level=log_level + 1,
            )

            for beta in it:
                it.update(f(beta=beta))
            # for beta in it.neighborhood:
            #     it.update(f(beta=beta))
            cost = it.y
        return cost

    def __call__(
        self,
        params: LWEParameters,
        success_probability: float = 0.99,
        red_cost_model=red_cost_model_default,
        log_level=1,
    ):
        """
        :param params: LWE parameters
        :param beta: Block size used to produce short dual vectors for dual reduction
        :param zeta: Dimension ζ ≥ 0 of new LWE instance
        :param h: Number of non-zero components of the secret of the new LWE instance
        :param success_probability: The success probability to target
        :param red_cost_model: How to cost lattice reduction
        """
        cost, h0 = Cost(rop=oo), -1
        while True:
            alternative = self.cost_h0(params, h0 + 1, success_probability, red_cost_model, log_level + 1)
            Logging.log("dual", log_level, f"h0={h0 + 1} --> {repr(alternative)}")
            if not alternative < cost:
                break
            cost = alternative
            h0 += 1
        return cost


chhs19 = CHHS19()


####################################################################################################
class DualHybridv2:
    """
    Estimate cost of solving LWE using dual attacks.
    Security level may be higher in practice, as this is optimistic for the attacker.
    """

    @staticmethod
    def _xi_factor(Xs, Xe):
        xi = RR(1)
        if Xs < Xe:
            xi = Xe.stddev / Xs.stddev
        return xi

    @staticmethod
    def sieving_cost(cost_model, beta):
        # convenience: instantiate static classes if needed
        if isinstance(cost_model, type):
            cost_model = cost_model()

        time = 2.0 ** (
            RR(cost_model.NN_AGPS[cost_model.nn]["a"]) * beta
            + RR(cost_model.NN_AGPS[cost_model.nn]["b"])
        )
        num_vectors = RR((4.0/3.0)**(beta / 2))
        return Cost(rop=time, red=time, beta=beta, mem=num_vectors)

    @staticmethod
    @cached_function
    def cost(
        beta: int,
        params: LWEParameters,
        num_targets: int,
        simulator=red_simulator_default,
        red_cost_model=red_cost_model_default,
        log_level=5,
    ):
        """
        Cost of the hybrid attack.

        :param beta: Block size.
        :param params: LWE parameters.
        :param zeta: Guessing dimension ζ ≥ 0.

        .. note :: This is the lowest level function that runs no optimization, it merely reports
           costs.

        """
        assert not params._homogeneous

        delta = deltaf(beta)
        d = min(ceil(sqrt(params.n * log(params.q) / log(delta))), params.n + params.m)
        xi = DualHybridv2._xi_factor(params.Xs, params.Xe)

        # 1. Simulate BKZ-β
        bkz_cost = costf(red_cost_model, beta, d)
        r = simulator(d, params.n, params.q, beta, xi=xi, tau=None, dual=True)

        # 2. Look at the lattice generate by the last block of length beta_sieve.
        # We want to balance the cost of BKZ and sieving. Note that BKZ makes a call to a sieve in
        # dimension beta - d4f(beta), but does this repeatedly (~8n times).
        # For simplicity, assume these two effects cancel out.
        beta_sieve = beta

        sieve_cost = DualHybridv2.sieving_cost(red_cost_model, beta_sieve)

        num_vectors = sieve_cost["mem"]
        k_fft = max(floor(log(num_vectors / num_targets)), 1)

        log_vol = sum(.5 * log(x, 2) for x in r[-beta_sieve:]) - log(2**k_fft)

        proj_length = params.Xe.stddev * sqrt(beta_sieve)
        proj_gh = exp(log_gh(beta_sieve, log_vol))

        if proj_length >= proj_gh:
            # It is impossible to distinguish the correct target from uniform targets.
            return Cost(rop=oo)

        if 2**k_fft * num_targets >= (proj_gh / proj_length)**beta_sieve:
            # We are in the concrete contradictory regime of [DP'23].
            # Thus, the attack will not work in expectation.
            return Cost(rop=oo)

        # cost = bkz_cost + sieve_cost
        cost = bkz_cost
        cost["rop"] += sieve_cost["rop"]
        cost["mem"] = RR(num_vectors * beta_sieve)

        # Time to compute the score for each targets.
        # Here, make simplifying assumption that we can run a FFT on all the targets without any
        # issues. Note: this is optimistic in general, as the structure doesn't allow direct
        # application of the FFT. This optimism includes the "modulus switching" technique
        # [MATZOV22], and pretends as if this technique doesn't increase the noise, which it does.

        cost["rop"] += RR(num_targets * (num_vectors + 2**k_fft * k_fft))

        # Success probability is definitely optimistic here!
        return cost.combine(Cost(prob=1.0))

    @classmethod
    def cost_zeta_(
        cls,
        params: LWEParameters,
        num_targets: int,
        simulator=red_simulator_default,
        red_cost_model=red_cost_model_default,
        log_level=5,
        **kwds,
    ):
        # Find smallest beta such that the correct solution is below GH of that block.
        def g(beta):
            return DualHybridv2.cost(
                beta, params, num_targets, simulator, red_cost_model, log_level, **kwds
            )

        def f(beta):
            return g(beta)["rop"] < oo

        d = params.n + params.m if params.m < oo else 2 * params.n
        opt_beta = 40 if f(40) else bisect_false_true(f, 40, d)
        cost = g(opt_beta)
        cost["|S|"] = num_targets
        return cost

    @classmethod
    def cost_zeta(
        cls,
        zeta: int,
        params: LWEParameters,
        simulator=red_simulator_default,
        red_cost_model=red_cost_model_default,
        log_level=5,
        **kwds,
    ):
        """
        This function optimizes costs for a fixed guessing dimension ζ.
        """
        if params.Xs.is_sparse:
            def cost_h(h):
                if h == 0:
                    # For h=0 perform custom code:
                    search_space = 1
                    reduced_params = params.updated(n=params.n - zeta)
                    prob = reduced_params.Xs.support_size() / params.Xs.support_size()
                else:
                    sea, red = params.Xs.split_balanced(zeta, h)
                    prob = params.Xs.split_probability(zeta, h)

                    search_space = sea.support_size()
                    reduced_params = params.updated(n=params.n - zeta, Xs=red)
                cost_ = cls.cost_zeta_(
                    reduced_params, search_space, simulator, red_cost_model, log_level, **kwds
                )

                if cost_["rop"] == oo:
                    return cost_
                cost_["prob"] *= prob
                cost_["h_"] = h
                return cost_.repeat(prob_amplify(0.99, prob))

            min_h = max(0, params.Xs.hamming_weight - (params.n - zeta))
            max_h = min(zeta, params.Xs.hamming_weight)

            cost = min(cost_h(h) for h in range(min_h, max_h + 1))
        else:
            # Non-sparse:
            search_space = params.updated(n=zeta).Xs.support_size()
            reduced_params = params.updated(n=params.n - zeta)
            cost = cls.cost_zeta_(
                reduced_params, search_space, simulator, red_cost_model, log_level, **kwds
            )

        cost["zeta"] = zeta
        return cost

    def __call__(
        self,
        params: LWEParameters,
        zeta: int = None,
        red_shape_model=red_shape_model_default,
        red_cost_model=red_cost_model_default,
        log_level=1,
        **kwds,
    ):
        """
        Estimate the cost of the dual hybrid attack.

        :param params: LWE parameters.
        :param zeta: Guessing dimension ζ ≥ 0.
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``red``: Number of word operations in lattice reduction.
        - ``δ``: Root-Hermite factor targeted by lattice reduction.
        - ``β``: BKZ block size.
        - ``ζ``: Number of guessed coordinates.
        - ``prob``: Probability of success in guessing.
        - ``repeat``: How often to repeat the attack.
        - ``d``: Lattice dimension.
        """
        params = LWEParameters.normalize(params)
        simulator = simulator_normalize(red_shape_model)

        f = partial(
            self.cost_zeta,
            params=params,
            simulator=simulator,
            red_cost_model=red_cost_model,
            log_level=log_level + 1,
        )

        if zeta is None:
            with local_minimum(1, params.n, log_level=log_level) as it:
                for zeta_ in it:
                    cost = f(zeta=zeta_, **kwds)
                    it.update(cost)
                    Logging.log("dual", log_level, f"ζ={zeta_}: {repr(cost)}")
                cost = it.y
        else:
            cost = f(zeta=zeta)

        cost["tag"] = "dual_hybrid_LB"
        return cost.sanity_check()

    __name__ = "dual_hybrid_LB"


dual_hybrid_v2 = DualHybridv2()
