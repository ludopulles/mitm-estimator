# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Estimate cost of solving LWE using primal attacks.

See :ref:`LWE Primal Attacks` for an introduction what is available.

"""
# from copy import copy
from functools import partial
from sage.all import oo, cached_function, ceil, exp, floor, log, pi, RR, sqrt

from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import red_simulator as red_simulator_default
from .cost import Cost
from .io import Logging
from .lwe_comb import log_comb, split_weight, sum_log
from .lwe_parameters import LWEParameters
from .lwe_primal import primal_usvp, PrimalUSVP, PrimalHybrid
from .nd import SparseTernary
from .prob import babai_gaussian, mitm_babai_probability, amplify as prob_amplify
from .reduction import cost as costf, delta as deltaf
from .simulator import normalize as simulator_normalize
from .util import local_minimum


class PrimalMeet:
    """
    Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
    """

    @staticmethod
    def cost_meet_lwe(search_space: SparseTernary, r: int, sigma: float, prob_adm: float):
        """
        Compute the cost of doing REP-1 depth 2 Meet-LWE for a given search space.

        :param search_space: search space to perform REP-1 over.
        :param r: simulated profile of reduced basis over which to meet.
        :param sigma: stddev of error (noise).

        """
        n = search_space.n
        n1, n2 = split_weight(n)

        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w = search_space.hamming_weight
        w0 = w // 2

        # Find the best epsilon by exhaustive search.
        # In practice, `epsilon` is 1, 2 or 3 so keep the range limited for performance.
        best_cost = Cost(rop=oo, prob=1.0)

        def cost_epsilon(epsilon):
            w1, w2 = split_weight(w0 + 2 * epsilon)  # Adds epsilon to each of w1 and w2.
            w11, w12 = split_weight(w1)
            w21, w22 = split_weight(w2)

            # Number of ways that we can construct s from s1 + s2.
            if n - w < 2 * epsilon or w0 < w2 - epsilon:
                return None

            # s has n-w zero entries:
            # `epsilon` come from 1 - 1, `epsilon` from -1 + 1, and rest from 0 + 0.
            # Half of the w0 +1's in s come from s1, and the other half from s2.
            # Similarly for the -1's.
            log_R1 = (log_comb(n - w, epsilon, epsilon)
                      + log_comb(w0, w1 - epsilon)
                      + log_comb(w0, w2 - epsilon))

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
            log_runtime = RR(sum_log(*logL.values()))

            # Analyse the success probability
            bet_s1 = logS[11] + logS[12] - logS[1]
            bet_s2 = logS[21] + logS[22] - logS[2]
            assert bet_s1 < 0 and bet_s2 < 0
            log_bet = bet_s1 + bet_s2

            # Logarithm of the number of buckets needed to have an efficient meet algorithm.
            # When using less, some buckets might get overcrowded, so listing near-collisions
            # becomes slow.
            log_idx = RR(max(logS[1], logS[2]))

            log_meet_leftover, dim_babai = log_idx, 0
            bound_meet = RR((len(r) * sigma)**2 * 2/pi)
            # Note: `r` contains *square norms*, so don't forget to take square roots...
            # r = copy(r)

            for i in range(len(r))[::-1]:
                dim_babai += 1
                if r[i] > bound_meet:
                    c_i = floor((r[i] / bound_meet)**.5)
                    # r[i] /= c_i**2
                    log_meet_leftover -= RR(log(c_i))
                    if log_meet_leftover < 0:
                        break

#            prob_adm = RR(mitm_babai_probability(r, sigma, fast=1000))

            # Multiply the runtime by the number of calls to Babai in dimension `d`.
            # And multiply the success probability with the admissibility probability.
            log_runtime += RR(log(PrimalHybrid.babai_cost(dim_babai)["rop"]))
#            log_bet += log(prob_adm)

            # Repeat the MEET algorithm 1/ (exp(log_bet) p_{adm}) times.
            # All these repetitions fail with probability of 1/e.
            runtime, prob = RR(exp(log_runtime - log_bet) / prob_adm), RR(1 - exp(-1))
            return Cost(
                rop=runtime, mem=runtime, prob=prob,
                h_1=w1, h_2=w11, epsilon=epsilon, log_idx=log_idx, dim_babai=dim_babai,
            )

        max_epsilon = 4
        # Compute costs for all `epsilon <= max_epsilon`, and pick the best.
        for epsilon in range(1, max_epsilon + 1):
            cost = cost_epsilon(epsilon)
            if not cost:
                break
            # As this Meet-LWE causes restarts on failure, minimize the Time/Probability ratio.
            if cost["rop"] < best_cost["rop"]:
                best_cost = cost
        # Try to improve the epsilon until a worse one is found
        while best_cost["epsilon"] == max_epsilon:
            max_epsilon += 1
            cost = cost_epsilon(max_epsilon)
            if cost["rop"] < best_cost["rop"]:
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
        d = min(max(beta, ceil(sqrt(n * log(params.q) / log(delta)))), params.m + n)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)

        # 1. Simulate BKZ-β
        sigma = params.Xe.stddev

        log_last_GS_norm = exp((log(params.q)*(d-n) + log(xi)*n) / d - (d-1) * log(delta))
        # if sqrt(r[-1]) < 10 * sigma:
        if log_last_GS_norm < 10 * sigma:
            # Lattice reduction should be sufficiently strong such that p_adm is of value >0.1 or so (e.g. 0.5).
            return Cost(rop=oo)
        r = simulator(d, n, params.q, beta, xi=xi, tau=None, dual=True)

        cost_bkz = RR(costf(red_cost_model, beta, d)["rop"])

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
            prob_hw = params.Xs.split_probability(zeta, hw)
            if prob_hw < 2**-20:
                # Very unlikely in this attack that the secret splits in this way.
                # The best attack has very small T/p (T = runtime, p = success probability)
                # The runtime is quite large so it also requires a somewhat large `p`.
                hw += 2
                continue

            # 3. Determine cost of doing Meet LWE.
            cost_meet = PrimalMeet.cost_meet_lwe(search_space, r, sigma, prob_adm)

            probability = (
                prob_hw  # prob. secret splits with given weights. Note: p_HW ~ 1/n roughly
                * prob_np  # prob. correct guess lifts with np. Note: p_NP ~ 1.0 (p_NP > p_adm).
                # * prob_adm  # prob. s = s1 + s2 has both (s1, s2) in the same bucket. (now part of MEET-LWE)
                * cost_meet["prob"]  # prob. Meet-LWE gives the correct answer.
            )

            cost = Cost({
                "rop": cost_bkz + cost_meet["rop"], "red": cost_bkz,
                "mem": cost_meet["mem"],
                "beta": beta, "zeta": zeta, "d": d,
                "h_": hw, "h_1": cost_meet["h_1"], "h_2": cost_meet["h_2"],
                "epsilon": cost_meet["epsilon"], "d_": cost_meet["dim_babai"],
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
        Cost.register_impermanent(h_1=False, h_2=False, d_=False, epsilon=False)

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
