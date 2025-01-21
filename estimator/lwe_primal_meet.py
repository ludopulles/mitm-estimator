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
from .lwe_comb import AsymptoticMeetREP1, log_comb
from .lwe_comb import split_weight, sum_log
from .lwe_parameters import LWEParameters
from .lwe_primal import primal_usvp, PrimalUSVP
from .nd import NoiseDistribution, SparseTernary
from .prob import babai_gaussian, mitm_babai_probability, amplify as prob_amplify
from .reduction import cost as costf, delta as deltaf
from .simulator import normalize as simulator_normalize
from .util import babai_cost, local_minimum


class PrimalMeet:
    """
    Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
    """
    _asymptotic = AsymptoticMeetREP1()

    @staticmethod
    def cost_meet_lwe(
        q: int,
        r,
        search_space: SparseTernary,
        Xe: NoiseDistribution
    ):
        n = search_space.n
        n1, n2 = split_weight(n)

        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w = search_space.hamming_weight
        w0 = w // 2
        w1, w2 = split_weight(w0)

        asymptotic_epsilon = int(round(n * PrimalMeet._asymptotic.optimal_epsilon(2, w0 / n)))
        w1 += asymptotic_epsilon
        w2 += asymptotic_epsilon

        w11, w12 = split_weight(w1)
        w21, w22 = split_weight(w2)

        # Number of ways that we can construct s from s1 + s2.
        log_R1 = (log_comb(n - w, asymptotic_epsilon, asymptotic_epsilon)
                  + log_comb(w0, w1 - asymptotic_epsilon)
                  + log_comb(w0, w2 - asymptotic_epsilon))

        # This code assumes that lattice reduction leaves some q-ary vectors at the start of the basis untouched, while
        # fully reducing the end of the basis (i.e. making its Gram--Schmidt norms LONGER).
        # This is slightly naive, but alternatively we could also meet keys somewhere in the intermediate part of
        # the basis. However, in that case, you need to take the mitm_babai_probability of the meet-buckets, instead of
        # the Babai domain.
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
        log_runtime += log(babai_cost(len(r)))

        # Analyse the success probability
        prob_rep_survives = mitm_babai_probability(r, Xe.stddev)
        bet_s1 = logS[11] + logS[12] - logS[1]
        bet_s2 = logS[21] + logS[22] - logS[2]
        assert bet_s1 < 0 and bet_s2 < 0

        log_bet = bet_s1 + bet_s2
        prob = prob_rep_survives * exp(log_bet)
        # print(f"MEET(2^{RR(log(prob_rep_survives, 2)):.1f}, 2^{RR(log_bet / log(2)):.1f}) ", end="")

        cost = Cost(
            rop=exp(log_runtime), mem=exp(log_runtime),
            h_1=w1, h_2=w11, epsilon=asymptotic_epsilon, ell=num_qary_vectors,
        )
        cost['r'] = r
        cost['tag'] = 'REP-0 (d=2)'
        return cost, prob

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
        delta = deltaf(beta)
        d = min(ceil(sqrt((params.n - zeta) * log(params.q) / log(delta))), params.m + (params.n - zeta) - 1)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)

        # 1. Simulate BKZ-β
        r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=None, dual=True)
        if r[-1] < 12 * params.Xe.stddev:
            # Lattice reduction should be sufficiently strong such that p_adm is of value >0.1 or so (e.g. 0.5).
            return Cost(rop=oo)

        cost_bkz = costf(red_cost_model, beta, d)

        # 2. Find the best hamming weight of the guess.
        # i.e. iteratively increase the HW until the number of guesses becomes too much.
        assert type(params.Xs) is SparseTernary
        assert params.Xs.p == params.Xs.m
        h, hw = params.Xs.hamming_weight, 0
        search_space = params.Xs.split_balanced(zeta, hw)[0]
        num_NP_available = RR(cost_bkz["rop"] / babai_cost(d))
        while hw + 2 <= min(h, zeta):
            new_search_space = params.Xs.split_balanced(zeta, hw + 2)[0]
            # Note: exponent 0.25 is very optimistic.
            # Searching all these keys will cost much more time...
            if RR(new_search_space.support_size()**0.35) >= num_NP_available:
                break
            search_space = new_search_space
            hw += 2

        if params.Xs.split_probability(zeta, hw) < 2**-20:
            # Very unlikely in this attack that the secret splits in this way.
            # The best attack has very small T/p (T = runtime, p = success probability)
            # In this case, the runtime is quite large, so it also requires a somewhat large `p`.
            return Cost(rop=oo)

        cost_meet, prob_meet = PrimalMeet.cost_meet_lwe(params.q, r, search_space, params.Xe)

        # Determine success probability:
        probability = (
            # p_HW: probability the secret splits its weight as such:
            # Note: p_HW ~ 1/n roughly
            params.Xs.split_probability(zeta, hw)
            # p_NP: probability that the correct guess lifts to the correct (s, e).
            # Note: p_NP ~ 1.0 (p_NP > p_adm).
            * RR(babai_gaussian(r, params.Xe.stddev))
            # p_MEET: probability that Meet-LWE gives the correct answer.
            * prob_meet
        )

        if not probability or RR(probability).is_NaN():
            return Cost(rop=oo)

        ret = Cost(
            {"|S|": search_space},
            rop=cost_bkz["rop"] + cost_meet["rop"],
            red=cost_bkz["rop"],
            mem=cost_meet["mem"],
            beta=beta, zeta=zeta, d=d, prob=probability,
            h_=hw, h_1=cost_meet["h_1"], h_2=cost_meet["h_2"],
            epsilon=cost_meet["epsilon"], ell=cost_meet["ell"],
        )

        # 4. Repeat whole experiment ~1/prob times
        ret = ret.repeat(prob_amplify(0.99, probability))

        # print(f"(β={beta}, ζ={zeta}): "
        #       f"probs {RR(params.Xs.split_probability(zeta, hw)):.8f}, "
        #       f"{RR(babai_gaussian(r, params.Xe.stddev)):.8f}, "
        #       f"{RR(prob_meet):.8f} and total runtime 2^{RR(log(ret['rop'], 2)):.1f}")
        return ret

    @classmethod
    def cost_zeta(
        self,
        zeta: int,
        params: LWEParameters,
        red_shape_model=red_simulator_default,
        red_cost_model=red_cost_model_default,
        optimize_d=True,
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
            self,
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
            for beta in it.neighborhood:
                it.update(f(beta))
            cost = it.y

        Logging.log("bdd", log_level, f"H1: {cost!r}")

        if cost is None:
            return Cost(rop=oo)
        return cost

    @classmethod
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
        Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
        """
        params = LWEParameters.normalize(params)
        red_shape_model = simulator_normalize(red_shape_model)
        Cost.register_impermanent(h_1=False, h_2=False, ell=False, epsilon=False)

        f = partial(
            self.cost_zeta,
            params=params,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            log_level=log_level + 1,
        )

        if zeta is None:
            # Find the smallest value for zeta such that the square root of the search space for
            # zeta is larger than the number of operations to solve uSVP on the whole LWE instance
            # (without guessing).
            usvp_cost = primal_usvp(params, red_cost_model=red_cost_model)["rop"]
            zeta_max = params.n
            while zeta_max < params.n and sqrt(params.Xs.resize(zeta_max).support_size()) < usvp_cost:
                zeta_max += 1

            with local_minimum(0, min(zeta_max, params.n), log_level=log_level) as it:
                for zeta in it:
                    it.update(f(zeta=zeta, optimize_d=False, **kwds))
            # TODO: this should not be required
            cost = min(it.y, f(0, optimize_d=False, **kwds))
        else:
            cost = f(zeta=zeta)

        cost["tag"] = "primal-meet-hybrid"
        cost["problem"] = params
        return cost.sanity_check()

    __name__ = "primal_meet"


primal_meet = PrimalMeet()
