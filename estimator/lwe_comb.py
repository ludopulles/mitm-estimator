# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using combinatorial attacks.

See :ref:`LWE Combinatorial Attacks` for an introduction what is available.
(TODO: write documentation on that page)
"""
from sage.all import binomial, exp, floor, log, oo, round, RR

from .cost import Cost
from .lwe_parameters import LWEParameters
from .nd import SparseTernary
from .prob import amplify as prob_amplify


def log_comb(n, *L):
    """
    Returns log(n! / (L[0]! L[1]! ... L[-1]!)), where sum(L) <= n and `log` is with base e.
    This is the number of ways to arrange `n` balls s.t. L[i] balls have colour i (i=0,1,...).
    EXAMPLE::

        >>> from estimator.lwe_comb import log_comb
        >>> log_comb(10, 3)
        4.78749...
        >>> log_comb(10, 3, 3)
        8.34283...

    """
    assert sum(L) <= n
    res = 0.0
    for x in L:
        res += log(binomial(n, x))
        n -= x
    return RR(res)


def sum_log(*L):
    """
    Return `log(sum(exp(x) for x in L))`, but without causing numerical issues.
    In particular, we use:
        `log(a_1 + ... + a_n)` = log(a_1) + log(1 + a_2/a_1 + ... + a_n/a_1)
                               = log(a_1) + log(sum(exp(log(a_i) - log(a_1)) for i=1,...,n)),
    when the input is `log(a_1), ..., log(a_n)`, and a_1 is maximal.
    """
    assert L
    if len(L) == 1:
        return L[0]

    max_L = max(L)
    return max_L + log(sum(exp(x - max_L) for x in L))


def split_weight(n):
    """
    Split up `n` evenly is: one equals `ceil(n / 2)`, and the other `floor(n / 2)`.
    :return: pair (a, b) such that a + b = n.
    """
    return (n + 1) // 2, n // 2


def bernoulli_success(log_p, log_n):
    """
    Return the probability of at least 1 success when having a bernoulli distribution with
    parameter `0 < p < 1`, and `n` many repetitions.
    """
    if log_p >= -20:
        return 1.0 - exp(exp(log_n) * log(1 - exp(log_p)))
    # For small `p`, we have 1 - (1-p)^n ~ 1 - e^{-np}.
    x = exp(log_n + log_p)
    if x < 1e-15:
        # For small `x`, we have 1 - e^{-x} ~ x - x^2 / 2 + x^3 / 6 + ...
        return x - x * x / 2.0
    return 1.0 - exp(-x)


class Odlyzko:
    def __call__(
        self,
        params: LWEParameters,
        target_probability=0.99,
        **kwds
    ):
        """
        Accurately estimate the cost of solving LWE using a MitM strategy dating back to Odlyzko.
        Note: this means we also compute the probability that a weight-w secret splits into two
        evenly weighted halves. This probability is >1/sqrt(n).

        :param params: the LWE parameters
        :param target_probability: the desired success probability of the attack
        :return: the cost to run this attack
        """
        # Check for ternary instead of sparse ternary.
        assert type(params.Xs) is SparseTernary, "Secret distribution has to be ternary."
        # Note: Odlyzko easily extends beyond "mean = 0
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, w0 = params.n, params.Xs.hamming_weight // 2

        nl, nr = split_weight(n)
        wl, wr = split_weight(w0)

        # Odlyzko splits s into s = (s_1 || s_2), where s_1 should be of weight wl and s_2 of
        # weight wr.
        log_S1, log_S2 = log_comb(nl, wl, wl), log_comb(nr, wr, wr)
        log_probability = log_S1 + log_S2 - log_comb(n, w0, w0)
        log_runtime = sum_log(log_S1, log_S2)
        repetitions = prob_amplify(target_probability, exp(log_probability))

        cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
        cost = cost.repeat(repetitions)
        cost['tag'] = 'Odlyzko'
        return cost


class MeetREP0:
    def __call__(
        self,
        params: LWEParameters,
        target_probability=0.99,
        **kwds,
    ):
        """
        Estimate cost of solving LWE using REP-0 from [C:May21]_.

        TODO: move this to the documentation:
        The goal is to recover the ternary secret `s` from b = As + e, where the secret is of a
        special form. Namely, `s \\in T^{n}(w/2)`, where this set `T^{n}(w/2)` consists of all
        ternary vectors with the number of 1's and -1's equal to `w/2`.

        The algorithm performs a Meet in the Middle (MitM) attack on s, by finding collisions (s1,
        s2) with (s1, s2) \\in T^{n}(w/4), such that:

            A s_1 + e_1 ~ A s_2 + e_2,    (1)

        where `e_1 \\in T^{n/2} x 0^{n/2}`, `e_2 \\in 0^{n/2} x T^{n/2}` (T = {-1,0,1}).

        Moreover, to construct s_1 and s_2 we also employ a MitM strategy such that both sides of
        (1) are close to a target `t` under a projection map onto `r` coefficients.

        :param params: the LWE parameters
        :param target_probability: the desired success probability of the attack
        :return: the cost to run this attack

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=200, q=127, Xs=ND.SparseTernary(10), Xe=ND.UniformMod(10))
            >>> LWE.meet_rep0(params)
            rop: ≈2^59.7, mem: ≈2^45.8, ↻: ≈2^13.9, r: 3, tag: REP-0 (d=2)

        """
        # Note: normalizing gives issues for the "GLP I" parameter set, so don't do that!
        # params = LWEParameters.normalize(params)

        # Check for ternary instead of sparse ternary.
        assert type(params.Xs) is SparseTernary, "Secret distribution has to be ternary."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, logq = params.n, RR(log(params.q))
        n1, n2 = split_weight(n)
        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1
        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w0 = params.Xs.hamming_weight // 2
        w1, w2 = split_weight(w0)
        w11, w12 = split_weight(w1)
        w21, w22 = split_weight(w2)

        # Number of ways that we can construct s from s1 + s2.
        log_R1 = 2 * log_comb(w0, w1)

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

        def optimize_r(r):
            # Warning: we make a bet on the weight distribution of:
            #     s1 = (s11 || s12), and s2 = (s21 || s22),
            # so we actually enumerate over LESS candidates s1 and s2!
            logL[1] = (logS[11] + logS[12]) - r * logq
            logL[2] = (logS[21] + logS[22]) - r * logq

            # Analyse the runtime
            log_time_guess = RR((r + 1) / 2) * log(error_width)
            log_time_lists = sum_log(*logL.values())
            log_runtime = log_time_guess + log_time_lists

            # Analyse the success probability
            prob_rep_survives = bernoulli_success(-r * logq, log_R1)
            bet_s1 = logS[11] + logS[12] - logS[1]
            bet_s2 = logS[21] + logS[22] - logS[2]
            assert bet_s1 < 0 and bet_s2 < 0

            log_bet = bet_s1 + bet_s2
            repetitions = prob_amplify(target_probability, prob_rep_survives * exp(log_bet))

            cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
            cost = cost.repeat(repetitions)
            cost['r'] = r
            cost['tag'] = 'REP-0 (d=2)'
            return cost

        # Find the optimal `r` here by iteratively incrementing `r`.
        best = {'rop': oo}
        r = floor(log_R1 / logq)
        while True:
            cost = optimize_r(r)
            if not cost <= best:
                break
            best = cost
            r += 1
        return best


class AsymptoticMeetREP1:
    """
    Methods to determine the epsilon parameters to run the Meet-LWE REP-1 attack optimally in the
    asymptotic setting.
    """
    BS_PRECISION = 20

    def H2(self, x):
        """
        Return `H(x, x, 1 - 2x)`, where H is the binary entropy function, base 2!
        """
        assert 0 < x < 0.5
        return -2 * x * log(x, 2) - (1.0 - 2 * x) * log(1.0 - 2 * x, 2)

    def binary_search_f(self, f, left, right):
        """
        Given a continuous increasing function f such that f(left) < 0 and f(right) > 0, return `x`
        such that f(x) = 0, using `BS_PRECISION` iterations.
        Absolute error: (right-left) / 2^{BS_PRECISION}.
        """
        for _ in range(self.BS_PRECISION):
            mid = 0.5 * (left + right)
            if f(mid) < 0:
                left = mid
            else:
                right = mid
        return 0.5 * (left + right)

    def ternary_search_f(self, f, left, right):
        """
        Given a continuous function f such that there is a local minimum `x` in [left, right], return
        `x`, using `BS_PRECISION` iterations.
        Absolute error: (right-left) * (2/3)^{BS_PRECISION}.
        """
        for _ in range(self.BS_PRECISION):
            a, b = (2 * left + right) / 3, (left + 2 * right) / 3
            if f(a) < f(b):
                right = b
            else:
                left = a
        return 0.5 * (left + right)

    def optimal_epsilon(self, depth, omega):
        """
        Tries to find the optimal parameter `epsilon`, which says that at level 1 of a tree of depth
        `depth`, you should have secrets with `omega/2 + eps` of them being 1's (and number of -1's),
        that combine into secrets of weight `omega` at level 0.

        :return: `epsilon`
        """
        def g(eps):
            base_time, sub_time = self.optimal_times(depth, omega, eps)
            return sub_time - base_time

        if depth == 2:
            # Base case: try to balance the cost of generating the two lists and the merged list.
            if g(0.0) > 0:
                # More time is already spent on the Odlyzko layer, so this is basically REP-0 (eps=0).
                return 0.0
            return self.binary_search_f(g, 0, 0.5 - omega)

        # First minimize the time spent on the base layer, by performing a ternary search.
        def f(eps):
            return self.optimal_times(depth, omega, eps)[0]

        opt_eps = self.ternary_search_f(f, 0, 0.5 - omega)
        if g(opt_eps) > 0:
            # Now, potentially find the sweet spot such that T_{base-layer} = T_{higher-layers},
            # where the maximum is equal to the two.
            return self.binary_search_f(g, 0, opt_eps)
        return opt_eps

    def optimal_times(self, depth, omega, eps):
        """
        Return the needed time for level 1, and deeper layers, by picking `delta` optimally.
        :param omega: a target weight for the layer below,
        :param eps: the increase of weight for the current layer.
        :param depth: depth of the tree
        :return: pair of 1) time for level 1 and 2) time for the lower layers
        """
        sub_omega = omega / 2 + eps
        R_1 = 2 * omega + (0 if eps == 0 else (1 - 2 * omega) * self.H2(eps / (1 - 2 * omega)))
        if depth == 2:
            # Base case.
            S_1 = self.H2(sub_omega)
            return S_1 - R_1, 0.5 * S_1

        delta = self.optimal_epsilon(depth - 1, sub_omega)
        sub_time = max(self.optimal_times(depth - 1, sub_omega, delta))

        S_2 = self.H2(sub_omega / 2 + delta)
        R_2 = 2 * sub_omega + (0 if delta == 0 else (1 - 2 * sub_omega) * self.H2(delta / (1 - 2 * sub_omega)))
        return 2 * S_2 - R_1 - R_2, sub_time


class MeetREP1:
    _asymptotic = AsymptoticMeetREP1()

    def depth2(
        self,
        params: LWEParameters,
        target_probability=0.99,
        **kwds,
    ):
        """
        Estimate cost of solving LWE using REP-1 with depth=2 from [C:May21]_.

        TODO: move this to the documentation:
        The goal is to recover the ternary secret `s` from b = As + e, where the secret is of a
        special form. Namely, `s \\in T^{n}(w/2)`, where this set `T^{n}(w/2)` consists of all
        ternary vectors with the number of 1's and -1's equal to `w/2`.

        The algorithm performs a Meet in the Middle (MitM) attack on s, by finding collisions (s1,
        s2) with (s1, s2) \\in T^{n}(w/4 + eps), such that:

            A s_1 + e_1 ~ A s_2 + e_2,    (1)

        where `e_1 \\in T^{n/2} x 0^{n/2}`, `e_2 \\in 0^{n/2} x T^{n/2}` (T = {-1,0,1}).

        Moreover, to construct s_1 and s_2 we also employ a MitM strategy such that both sides of
        (1) are close to a target `t` under a projection map onto `r` coefficients.

        :param params: the LWE parameters
        :param target_probability: the desired success probability of the attack
        :return: the cost to run this attack

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=200, q=127, Xs=ND.SparseTernary(10), Xe=ND.UniformMod(10))
            >>> LWE.meet_rep1(params)
            rop: ≈2^59.1, mem: ≈2^50.3, ↻: 452, r: 4, ε: 1, tag: REP-1 (d=2)

        """
        # Note: normalizing gives issues for the "GLP I" parameter set, so don't do that!
        # params = LWEParameters.normalize(params)

        # Check for ternary instead of sparse ternary.
        assert type(params.Xs) is SparseTernary, "Secret distribution has to be ternary."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, logq = params.n, RR(log(params.q))
        n1, n2 = split_weight(n)
        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1
        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w = params.Xs.hamming_weight
        w0 = w // 2
        w1, w2 = split_weight(w0)

        asymptotic_epsilon = int(round(n * self._asymptotic.optimal_epsilon(2, w0 / n)))
        w1 += asymptotic_epsilon
        w2 += asymptotic_epsilon

        w11, w12 = split_weight(w1)
        w21, w22 = split_weight(w2)

        # Number of ways that we can construct s from s1 + s2.
        log_R1 = (log_comb(n - w, asymptotic_epsilon, asymptotic_epsilon)
                  + log_comb(w0, w1 - asymptotic_epsilon)
                  + log_comb(w0, w2 - asymptotic_epsilon))

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

        def optimize_r(r):
            # Warning: we make a bet on the weight distribution of:
            #     s1 = (s11 || s12), and s2 = (s21 || s22),
            # so we actually enumerate over LESS candidates s1 and s2!
            logL[1] = (logS[11] + logS[12]) - r * logq
            logL[2] = (logS[21] + logS[22]) - r * logq

            # Analyse the runtime
            log_time_guess = RR((r + 1) / 2) * log(error_width)
            log_time_lists = sum_log(*logL.values())
            log_runtime = log_time_guess + log_time_lists

            # Analyse the success probability
            prob_rep_survives = bernoulli_success(-r * logq, log_R1)
            bet_s1 = logS[11] + logS[12] - logS[1]
            bet_s2 = logS[21] + logS[22] - logS[2]
            assert bet_s1 < 0 and bet_s2 < 0

            log_bet = bet_s1 + bet_s2
            repetitions = prob_amplify(target_probability, prob_rep_survives * exp(log_bet))

            cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
            cost = cost.repeat(repetitions)
            cost['r'] = r
            cost['epsilon'] = asymptotic_epsilon
            cost['tag'] = 'REP-1 (d=2)'
            return cost

        # Find the optimal `r` here by iteratively incrementing `r`.
        best = {'rop': oo}
        r = floor(log_R1 / logq)
        while True:
            cost = optimize_r(r)
            if not cost <= best:
                break
            best = cost
            r += 1
        return best

    def depth3(
        self,
        params: LWEParameters,
        target_probability=0.99,
        **kwds,
    ):
        """
        Estimate cost of solving LWE using REP-1 with depth=3 from [C:May21]_.

        :param params: the LWE parameters
        :param target_probability: the desired success probability of the attack
        :return: the cost to run this attack

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.
        """
        # Note: normalizing gives issues for the "GLP I" parameter set, so don't do that!
        # params = LWEParameters.normalize(params)

        # Check for ternary instead of sparse ternary.
        assert type(params.Xs) is SparseTernary, "Secret distribution has to be ternary."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, logq = params.n, RR(log(params.q))
        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1

        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w = params.Xs.hamming_weight
        w0 = w // 2

        def optimize_eps(eps1, eps2):
            w1 = (w0 + 1) // 2 + eps1
            w2 = (w1 + 1) // 2 + eps2

            # Analyse level 3 (using Odlyzko):
            log_S3 = log_comb((n + 1) // 2, *split_weight(w2))
            log_T3 = log_S3  # Time for enumerating over whole search space S3.

            # Analyse level 2 (using Howard-Graham):
            log_R2 = log_comb(n - 2 * w1, eps2, eps2) + 2 * log_comb(w1, w1 // 2)
            log_S2 = log_comb(n, w2, w2)
            log_T2 = 2 * log_S3 - log_R2  # Time for building lists in level 2 from Odlyzko.

            # Analyse level 1 (using Howard-Graham):
            log_R1 = log_comb(n - 2 * w0, eps1, eps1) + 2 * log_comb(w0, w0 // 2)
            log_T1 = 2 * log_S2 - log_R1 - log_R2  # Time for building the lists in level 1

            log_time_lists = sum_log(
                log_T1 + log(2),  # 2 * T1
                log_T2 + log(4),  # 4 * T2
                log_T3 + log(8),  # 8 * T3
            )

            # Take `r_1, r_2` giving a probability >= 1/e that a representation survives.
            r1 = floor(log_R1 / logq)
            r2 = floor(log_R2 / logq)

            # Analyse the runtime
            log_time_guess = RR((r1 + 1) / 2) * log(error_width)
            log_runtime = log_time_guess + log_time_lists

            # Probability that the correct solution is found. When taking the floor for r_1, this
            # should be >= 1/e, and very close to 1 in practice.
            prob_rep_survives = bernoulli_success(-r1 * logq, log_R1)
            # Probability that the weight splits equally over the two halfs, during Odlyzko.
            # Note: Stirling's approximation yields:
            # exp(log_bet) ~ 2/pi * sqrt(n / (w0 * w0 * (n - 2*w0))) >= 3 / n.
            log_bet = 2 * log_comb(n // 2, w0 // 2, w0 // 2) - log_comb(n, w0, w0)
            repetitions = prob_amplify(target_probability, prob_rep_survives * exp(log_bet))

            cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
            cost = cost.repeat(repetitions)

            # cost += {'T_1': exp(log_T1), 'T_2': exp(log_T2), 'T_3': exp(log_T3)}
            cost += {'r_1': r1, 'r_2': r2, 'epsilon_1': eps1, 'epsilon_2': eps2}
            cost['tag'] = 'REP-1 (d=3)'
            return cost

        eps1 = int(round(n * self._asymptotic.optimal_epsilon(3, w0 / 2 / n)))
        eps2 = int(round(n * self._asymptotic.optimal_epsilon(2, (w0 / 2 + eps1) / 2 / n)))
        best = optimize_eps(eps1, eps2)
        # Find the optimal `r` here by iteratively incrementing `eps_1` and/or `eps_2`.
        while True:
            # Also try incrementing eps1 by 2, because it changes w2 after rounding down.
            # This ordering gave the best results when running on NTRU (Prime), BLISS, and GLP.
            for (delta1, delta2) in [(1, 0), (2, 0), (0, 1)]:
                cost = optimize_eps(eps1 + delta1, eps2 + delta2)
                if cost < best:
                    best = cost
                    eps1 += delta1
                    eps2 += delta2
                    break
            else:
                # No improvement, stop.
                break
        return best

    def __call__(
        self,
        params: LWEParameters,
        target_probability=0.99,
        **kwds,
    ):
        d_2 = self.depth2(params, target_probability, *kwds)
        d_3 = self.depth3(params, target_probability, *kwds)
        return min(d_2, d_3)


odlyzko = Odlyzko()
meet_rep0 = MeetREP0()
meet_rep1 = MeetREP1()
