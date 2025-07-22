# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using primal attacks.

See :ref:`LWE Primal Attacks` for an introduction what is available.

"""
from functools import partial
from sage.all import oo, cached_function, ceil, exp, log, RR, sqrt

from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import red_simulator as red_simulator_default
from .cost import Cost
from .io import Logging
from .lwe_comb import log_comb, split_weight, sum_log
from .lwe_parameters import LWEParameters
from .lwe_primal import primal_usvp, PrimalUSVP
from .nd import SparseTernary
from .prob import babai_gaussian, mitm_babai_probability, amplify as prob_amplify
from .reduction import cost as costf, delta as deltaf
from .simulator import normalize as simulator_normalize
from .util import babai_cost, local_minimum


class PrimalMeet:
    """
    Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
    """

    @staticmethod
    def cost_meet_lwe(q: int, d: int, search_space: SparseTernary):
        """
        Compute the cost of doing REP-1 depth 2 Meet-LWE for a given search space.
        """
        n = search_space.n
        n1, n2 = split_weight(n)

        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w = search_space.hamming_weight
        w0 = w // 2

        # Find the best epsilon by exhaustive search.
        # In practice, `epsilon` is 1, 2 or 3 so keep the range limited for performance.
        best_cost = Cost(rop=oo, prob=1.0)
        for epsilon in range(1, 5):
            w1, w2 = split_weight(w0 + 2 * epsilon)  # Adds epsilon to each of w1 and w2.
            w11, w12 = split_weight(w1)
            w21, w22 = split_weight(w2)

            # Number of ways that we can construct s from s1 + s2.
            if n - w < 2 * epsilon or w0 < w2 - epsilon:
                break
            log_R1 = (log_comb(n - w, epsilon, epsilon)
                      + log_comb(w0, w1 - epsilon)
                      + log_comb(w0, w2 - epsilon))

            # This code assumes that lattice reduction leaves some q-ary vectors at the start of the basis untouched,
            # while fully reducing the end of the basis (i.e. making its Gram--Schmidt norms LONGER).
            # This is slightly naive, but alternatively we could also meet keys somewhere in the intermediate part of
            # the basis. However, in that case, you need to take the mitm_babai_probability of the meet-buckets,
            # instead of the Babai domain.
            logq = RR(log(q))
            num_qary_vectors = ceil(log_R1 / logq)
            # assert log_R1 < logq

            logS = {}  # log(size of search space)
            logL = {}  # log(size of list of stored candidates)

            # Analyse level 2.
            logL[11] = logS[11] = log_comb(n1, w11, w11)
            logL[12] = logS[12] = log_comb(n2, w12, w12)
            logL[21] = logS[21] = log_comb(n1, w21, w21)
            logL[22] = logS[22] = log_comb(n2, w22, w22)

            # Analyse level 1.
            logS[1] = log_comb(n, w1, w1)
            logS[2] = log_comb(n, w2, w2)
            logL[1] = logS[1] - log_R1
            logL[2] = logS[2] - log_R1

            # Analyse the runtime
            log_runtime = sum_log(*logL.values())
            # Multiply the runtime by the number of calls to Babai in dimension `d`.
            log_runtime += log(babai_cost(d))

            # Analyse the success probability
            bet_s1 = logS[11] + logS[12] - logS[1]
            bet_s2 = logS[21] + logS[22] - logS[2]
            assert bet_s1 < 0 and bet_s2 < 0
            log_bet = bet_s1 + bet_s2

            cost = Cost(
                rop=RR(exp(log_runtime)), mem=RR(exp(log_runtime)), prob=RR(exp(log_bet)),
                h_1=w1, h_2=w11, epsilon=epsilon, ell=num_qary_vectors
            )
            # As this Meet-LWE causes restarts on failure, minimize the Time/Probability ratio.
            if cost["rop"] / cost["prob"] < best_cost["rop"] / best_cost["prob"]:
                best_cost = cost
        return best_cost

    @staticmethod
    @cached_function
    def cost(
        params: LWEParameters,
        beta: int,
        zeta: int,
        simulator=red_simulator_default,
        red_cost_model=red_cost_model_default,
        log_level=5,
    ):
        """
        Cost of the Primal Meet-LWE attack.

        :param beta: Block size.
        :param params: LWE parameters.
        :param zeta: Guessing dimension ζ ≥ 0.
        :param m: We accept the number of samples to consider from the calling function.

        .. note :: This is the lowest level function that runs no optimization. It merely reports costs.

        """
        assert isinstance(params.Xs, SparseTernary) and params.Xs.p == params.Xs.m

        delta = deltaf(beta)
        n = params.n - zeta
        d = min(ceil(sqrt(n * log(params.q) / log(delta))), params.m + n)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)

        # 1. Simulate BKZ-β
        sigma = params.Xe.stddev

        log_last_GS_norm = exp((log(params.q)*(d-n) + log(xi)*n) / d - (d-1) * log(delta))
        # if sqrt(r[-1]) < 10 * sigma:
        if log_last_GS_norm < 10 * sigma:
            # Lattice reduction should be sufficiently strong such that p_adm is of value >0.1 or so (e.g. 0.5).
            return Cost(rop=oo)
        r = simulator(d, n, params.q, beta, xi=xi, tau=None, dual=True)

        cost_bkz = costf(red_cost_model, beta, d)
        num_np_available = RR(cost_bkz["rop"] / babai_cost(d))

        prob_np = RR(babai_gaussian(r, sigma))
        prob_adm = RR(mitm_babai_probability(r, sigma, fast=1000))
        # print(f"zeta={zeta}, beta={beta} and r[-1]={float(r[-1] / sigma):6.1g} "
        #       f"-> {float(RR(mitm_babai_probability(r, sigma))):.5f} vs {float(prob_adm):.5f}")

        # 2. Find the best hamming weight of the guess.
        best_cost = Cost(rop=oo)

        # i.e. iteratively increase the HW until the number of guesses becomes too much.
        h, hw = params.Xs.hamming_weight, 0
        while hw <= min(h, zeta):
            search_space = params.Xs.split_balanced(zeta, hw)[0]
            # Note: exponent 0.25 is very optimistic.
            # Searching all these keys will cost much more time...
            if RR(search_space.support_size()**0.25) >= num_np_available:
                break

            prob_hw = params.Xs.split_probability(zeta, hw)
            if prob_hw < 2**-20:
                # Very unlikely in this attack that the secret splits in this way.
                # The best attack has very small T/p (T = runtime, p = success probability)
                # The runtime is quite large so it also requires a somewhat large `p`.
                hw += 2
                continue

            # 3. Determine cost of doing Meet LWE.
            cost_meet = PrimalMeet.cost_meet_lwe(params.q, d, search_space)

            probability = (
                prob_hw  # prob. secret splits with given weights. Note: p_HW ~ 1/n roughly
                * prob_np  # prob. correct guess lifts with np. Note: p_NP ~ 1.0 (p_NP > p_adm).
                * prob_adm  # prob. s = s1 + s2 has both (s1, s2) in the same bucket.
                * cost_meet["prob"]  # prob. Meet-LWE gives the correct answer.
            )

            cost = Cost({
                "rop": cost_bkz["rop"] + cost_meet["rop"], "red": cost_bkz["rop"],
                "mem": cost_meet["mem"],
                "beta": beta, "zeta": zeta, "d": d,
                "h_": hw, "h_1": cost_meet["h_1"], "h_2": cost_meet["h_2"],
                "epsilon": cost_meet["epsilon"], "ell": cost_meet["ell"],
                "|S|": search_space, "prob": probability,
            })

            # 4. Repeat whole experiment ~1/prob times
            # assert not (not probability or RR(probability).is_NaN())
            cost = cost.repeat(prob_amplify(0.99, probability))

            if cost > best_cost:
                break
            best_cost = min(best_cost, cost)
            hw += 2
        return best_cost

    @classmethod
    def cost_zeta(
        cls,
        zeta: int,
        params: LWEParameters,
        red_shape_model=red_simulator_default,
        red_cost_model=red_cost_model_default,
        log_level=5,
        **kwds,
    ):
        """
        This function optimizes costs for a fixed guessing dimension ζ.
        """

        # step 0. establish baseline
        baseline_cost = primal_usvp(
            params,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            optimize_d=False,
            log_level=log_level + 1,
            **kwds,
        )
        Logging.log("bdd", log_level, f"H0: {repr(baseline_cost)}")

        f = partial(
            PrimalMeet.cost,
            params,
            zeta=zeta,
            simulator=red_shape_model,
            red_cost_model=red_cost_model,
            **kwds,
        )

        # step 1. optimize β
        with local_minimum(
            40, baseline_cost["beta"] + 1, precision=2, log_level=log_level + 1
        ) as it:
            for beta in it:
                it.update(f(beta))
            cost = it.y

        Logging.log("bdd", log_level, f"H1: {cost!r}")

        if cost is None:
            return Cost(rop=oo)
        return cost

    @classmethod
    def __call__(
        cls,
        params: LWEParameters,
        zeta: int = None,
        red_shape_model=red_shape_model_default,
        red_cost_model=red_cost_model_default,
        log_level=1,
        **kwds,
    ):
        """
        Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
        """
        params = LWEParameters.normalize(params)
        red_shape_model = simulator_normalize(red_shape_model)
        Cost.register_impermanent(h_1=False, h_2=False, ell=False, epsilon=False)

        f = partial(
            cls.cost_zeta,
            params=params,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            log_level=log_level + 1,
        )

        if zeta is None:
            cost = Cost(rop=oo)
            with local_minimum(0, params.n, precision=2, log_level=log_level) as it:
                for zeta_ in it:
                    it.update(f(zeta=zeta_, **kwds))
                cost = it.y
        else:
            cost = f(zeta=zeta)

        cost["tag"] = "primal-meet-hybrid"
        cost["problem"] = params
        return cost.sanity_check()

    __name__ = "primal_meet"


primal_meet = PrimalMeet()
