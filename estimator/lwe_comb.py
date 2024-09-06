# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using combinatorial attacks.

See :ref:`LWE Combinatorial Attacks` for an introduction what is available.
(TODO: write documentation on that page)
"""
from sage.all import binomial, exp, floor, log, oo, RR

from .cost import Cost
from .lwe_parameters import LWEParameters
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
        log_level=1,
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
        assert params.Xs.tag == "SparseTernary", "Secret distribution has to be ternary."
        # Note: Odlyzko easily extends beyond "mean = 0
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, w0 = params.n, params.Xs.get_hamming_weight(params.n) // 2

        nl, nr = split_weight(n)
        wl, wr = split_weight(w0)

        # Odlyzko splits s into s = (s_1 || s_2), where s_1 should be of weight wl and s_2 of
        # weight wr.
        log_S1, log_S2 = log_comb(nl, wl, wl), log_comb(nr, wr, wr)
        log_probability = log_S1 + log_S2 - log_comb(n, w0, w0)
        log_runtime = sum_log(log_S1, log_S2)
        repetitions = prob_amplify(0.99, exp(log_probability))

        cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
        cost.register_impermanent(rop=True, mem=False)
        cost = cost.repeat(repetitions)
        cost["tag"] = 'Odlyzko'
        return cost


class MeetRep0:
    def __call__(
        self,
        params: LWEParameters,
        target_probability=0.99,
        log_level=1,
        **kwds,
    ):
        """
        Estimate cost of solving LWE using REP-0 from [May21]_.

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

        :param params: LWE parameters.
        :param target_probability: the desired lower bound on the probability that (a
        number of repetitions of) the algorithm yields the correct output.
        :return: A cost dictionary.

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=200, q=127, Xs=ND.SparseTernary(200, 10), Xe=ND.UniformMod(10))
            >>> LWE.meet_rep0(params)
            rop: ≈2^58.2, mem: ≈2^49.3, ↻: 476, tag: REP-0

        """
        # Note: normalizing gives issues for the "GLP I" parameter set, so don't do that!
        # params = LWEParameters.normalize(params)

        # Check for ternary instead of sparse ternary.
        assert params.Xs.tag == "SparseTernary", "Secret distribution has to be ternary."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        n, logq = params.n, RR(log(params.q))
        n1, n2 = split_weight(n)
        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1
        # The secret has `w0` coefficients equal to 1, and `w0` equal to -1.
        w0 = params.Xs.get_hamming_weight(n) // 2
        w1, w2 = split_weight(w0)
        w11, w12 = split_weight(w1)
        w21, w22 = split_weight(w2)

        # number of ways to write s = s_1 + s_2
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

        # Number of ways that we can construct s from s1 + s2.
        log_R1 = log_comb(w0, w1) * 2.0

        # Find the optimal `r` here by iteratively incrementing `r`.
        r = floor(log_R1 / logq)
        cur_cost = {'rop': oo}
        while True:
            # Warning: we make a bet on the weight distribution of:
            #     s1 = (s11 || s12), and s2 = (s21 || s22),
            # so we actually enumerate over LESS candidates s1 and s2!
            logL[1] = (logS[11] + logS[12]) - r * logq
            logL[2] = (logS[21] + logS[22]) - r * logq

            # Analyse the runtime
            log_time_guess = RR(((r + 1) // 2) * log(error_width))
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
            cost.register_impermanent(rop=True, mem=False)
            cost = cost.repeat(repetitions)
            cost["tag"] = 'REP-0'

            if not cost <= cur_cost:
                break
            cur_cost = cost
            r += 1
        return cur_cost


odlyzko = Odlyzko()
meet_rep0 = MeetRep0()
