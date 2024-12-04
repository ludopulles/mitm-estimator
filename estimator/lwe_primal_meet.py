# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using primal attacks.

See :ref:`LWE Primal Attacks` for an introduction what is available.

"""
from functools import partial

from sage.all import oo, cached_function, ceil, floor, log, RR, sqrt
from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import red_simulator as red_simulator_default
from .cost import Cost
from .io import Logging
from .lwe_comb import AsymptoticMeetREP1, log_comb
from .lwe_parameters import LWEParameters
from .lwe_primal import primal_usvp, PrimalUSVP, PrimalHybrid
from .nd import NoiseDistribution, SparseTernary
from .prob import amplify as prob_amplify
from .prob import babai_gaussian
from .reduction import cost as costf
from .reduction import delta as deltaf
from .simulator import normalize as simulator_normalize
from .util import local_minimum


class PrimalMeet(PrimalHybrid):
    """
    Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
    """
    _asymptotic = AsymptoticMeetREP1()

    @cached_function
    def analyze_LSH(
        self,
        r,
        ell: float,
        stddev: float,
    ):
        ns = [floor(ri / ell) for ri in r]
        lengths = [ri / ni for ri, ni in zip(r, ns)]
        collide_prob = prod(ni for ni in ns)
        mitm_prob = mitm_babai_probability(r, stddev)

    @cached_function
    def cost_meet_lwe(
        self,
        r,
        search_space: SparseTernary,
        Xe: NoiseDistribution
    ):
        # Figure 6. Parameter search range
        ell = 6 * Xe.stddev
        b_lsh = 2 * ell
        zeta = search_space.n
        hw = search_space.hamming_weight

        time = 0
        prob = 1.0

        w0 = hw//2  # Expected number of 1's. Equals expected number of -1's.
        eps1 = int(round(zeta * self._asymptotic.optimal_epsilon(2, w0 / zeta)))
        w1 = w0 + eps1

        # Number of ways that we can construct s from s1 + s2.
        log_R = RR(log_comb(zeta - hw, eps1, eps1) + 2 * log_comb(w0, w1 - eps1))

        # Split s = s1 + s2.
        # Split s1, s2 Odlyzko-style.
        # s1, s2 each have w1 +1's and w1 -1's.
        # Perhaps, search-space size of S1, S2 is already small enough? Do we need more filtering?
        # If so, what do we condition on?

        # Base on https://github.com/yonghaason/PrimalMeetLWE/blob/main/estimator/estimator.py
        # Is it correct?

        return time, prob

    @cached_function
    def cost(
        self,
        params: LWEParameters,
        beta: int,
        zeta: int,
        m: int = oo,
        d: int = None,
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
        :param d: We optionally accept the dimension to pick.

        .. note :: This is the lowest level function that runs no optimization, it merely reports
           costs.

        """
        if d is None:
            delta = deltaf(beta)
            d = min(ceil(sqrt(params.n * log(params.q) / log(delta))), m) + 1
        d -= zeta

        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)
        tau = 1  # TODO: pick τ as non default value
        if params._homogeneous:
            tau = False
            d -= 1

        # 1. Simulate BKZ-β
        r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=tau, dual=True)
        cost_bkz = costf(red_cost_model, beta, d)
        cost_svp = PrimalMeet.babai_cost(d)

        # 2. Find the best hamming weight of the guess.
        # i.e. iteratively increase the HW until the number of guesses becomes too much.
        assert type(params.Xs) is SparseTernary
        assert params.Xs.p == params.Xs.m
        h, hw = params.Xs.hamming_weight, 0
        search_space = params.Xs.split_balanced(zeta, hw)[0]
        while hw + 2 <= min(h, zeta):
            new_search_space = params.Xs.split_balanced(zeta, hw + 2)[0]
            # Note: exponent 0.25 is very optimistic.
            # Searching all these keys will cost much more time...
            if cost_svp.repeat(RR(new_search_space.support_size()**0.25))["rop"] >= cost_bkz["rop"]:
                break
            search_space = new_search_space
            hw += 2

        r0 = -0
        cost_meet, prob_meet = self.cost_meet_lwe(r[-r0:], search_space)
        # prob_meet = mitm_babai_probability(r, params.Xe.stddev)

        # Determine success probability:
        probability = (
            # p_HW: probability the secret splits its weight as such:
            params.Xs.split_probability(zeta, hw)
            # p_NP: probability that the correct guess lifts to the correct (s, e).
            * RR(babai_gaussian(r, params.Xe.stddev))
            # p_MEET: probability that Meet-LWE gives the correct answer.
            * prob_meet
        )

        ret = Cost(
            rop=cost_bkz["rop"] + cost_meet,
            red=cost_bkz["rop"],
            beta=beta, zeta=zeta, d=d, prob=probability
        )
        ret["|S|"] = search_space

        # 4. Repeat whole experiment ~1/prob times
        if not probability or RR(probability).is_NaN():
            return Cost(rop=oo)
        return ret.repeat(prob_amplify(0.99, probability))

    @classmethod
    def cost_zeta(
        self,
        zeta: int,
        params: LWEParameters,
        red_shape_model=red_simulator_default,
        red_cost_model=red_cost_model_default,
        m: int = oo,
        babai: bool = True,
        mitm: bool = True,
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
            self.cost,
            params=params,
            zeta=zeta,
            babai=babai,
            mitm=mitm,
            simulator=red_shape_model,
            red_cost_model=red_cost_model,
            m=m,
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

        # step 2. optimize d
        if cost and cost.get("tag", "XXX") != "usvp" and optimize_d:
            with local_minimum(
                params.n, cost["d"] + cost["zeta"] + 1, log_level=log_level + 1
            ) as it:
                for d in it:
                    it.update(f(beta=cost["beta"], d=d))
                cost = it.y
            Logging.log("bdd", log_level, f"H2: {cost!r}")

        if cost is None:
            return Cost(rop=oo)
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
        Estimate cost of solving LWE via a hybrid of primal attack and Meet-LWE [HKLS22].
        """
        params = LWEParameters.normalize(params)
        red_shape_model = simulator_normalize(red_shape_model)
        Cost.register_impermanent(
            {"|S|": False}, rop=True, red=True, svp=True, zeta=False, prob=False,
        )

        # allow for a larger embedding lattice dimension: Bai and Galbraith
        m = params.m + params.n if params.Xs <= params.Xe else params.m

        f = partial(
            self.cost_zeta,
            params=params,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            m=m,
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
