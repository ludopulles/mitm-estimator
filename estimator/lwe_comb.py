# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using primal attacks.

See :ref:`LWE Primal Attacks` for an introduction what is available.

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


class Odlyzko:
    def __call__(
        self,
        params: LWEParameters,
        log_level=1,
        **kwds
    ):
        """
        Estimate cost of solving LWE using a MitM strategy dating back to Odlyzko.
        Note: when comparing to [May21], they use |S|^{0.5} instead of an actual computation,
              which should actually be the same up to polynomial factors.

        :param params: LWE parameters.
        :return: A cost dictionary.

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.
        """

        # Check for ternary instead of sparse ternary.
        assert params.Xs.tag == "SparseTernary", "Secret distribution has to be ternary."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."

        # Note: Odlyzko easily extends beyond "mean = 0
        assert params.Xs.mean == 0, "Expected #1's == #-1's."

        n = params.n

        # Note: out of the `n` secret coefficients, `weight_s` are +1 and `weight_s` are -1.
        w = params.Xs.get_hamming_weight(n) // 2

        nl, nr = split_weight(n)
        wl, wr = split_weight(w)

        log_S1, log_S2 = log_comb(nl, wl, wl), log_comb(nr, wr, wr)
        log_probability = log_S1 + log_S2 - log_comb(n, w, w)
        log_runtime = sum_log(log_S1, log_S2)
        repetitions = prob_amplify(0.99, exp(log_probability))

        cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
        cost.register_impermanent(rop=True, mem=False)
        cost = cost.repeat(repetitions)
        cost["tag"] = "Odlyzko MitM"
        return cost


class MeetRep0:
    def __call__(
        self,
        params: LWEParameters,
        target_success_probability=0.99,
        log_level=1,
        **kwds,
    ):
        """
        Estimate cost of solving LWE via Meet-LWE Rep-0 [May21]_.

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
        :param target_success_probability: the desired lower bound on the probability that (a
        number of repetitions of) the algorithm yields the correct output.
        :return: A cost dictionary.

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=200, q=127, Xs=ND.SparseTernary(200, 10), Xe=ND.UniformMod(10))
            >>> LWE.meet_rep0(params)
            rop: ≈2^58.2, mem: ≈2^49.3, ↻: 476, tag: [May21]

        """
        # Note: normalizing gives issues for the "GLP I" parameter set, so don't do that!
        # params = LWEParameters.normalize(params)

        # Check for ternary instead of sparse ternary.
        assert params.Xs.tag == "SparseTernary", "Secret distribution has to be ternary."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."

        n, m, logq = params.n, params.m, RR(log(params.q))
        if params.m == oo:
            m = n

        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1

        logS = {}  # log(size of search space)
        logR = {}  # log(number of representatives) (i.e. ways to write s = s1 + s2)
        logL = {}  # log(size of list of stored candidates)

        # Note: out of the `n` secret coefficients, `weight_s` are +1 and `weight_s` are -1.
        weight_s = params.Xs.get_hamming_weight(n) // 2

        weight_s1, weight_s2 = split_weight(weight_s)

        weight_s11, weight_s12 = split_weight(weight_s1)
        weight_s21, weight_s22 = split_weight(weight_s2)
        n_left, n_right = split_weight(n)

        # Analyse level 2.
        logL[11] = logS[11] = log_comb(n_left, weight_s11, weight_s11)
        logL[12] = logS[12] = log_comb(n_right, weight_s12, weight_s12)
        logL[21] = logS[21] = log_comb(n_left, weight_s21, weight_s21)
        logL[22] = logS[22] = log_comb(n_right, weight_s22, weight_s22)

        # Analyse level 1.
        logS[1] = log_comb(n, weight_s1, weight_s1)
        logS[2] = log_comb(n, weight_s2, weight_s2)

        # Number of ways that we can construct s from s1 + s2.
        # There is flexibility where the w_s = w_s1 + w_s2 +1's come from. Similarly for -1's.
        logR[1] = log_comb(weight_s, weight_s1) * 2.0

        # Note: `r = floor(logR[1] / logq)` is asymptotically optimal,
        # and lower bounds the probability a representation `s = s1 + s2` survives by 1/e.
        # However, experimentally, it can sometimes be better to increase `r` by 1 or 2.
        # TODO: find the optimal `r` here by iteratively incrementing `r`.
        r = floor(logR[1] / logq)

        # Warning: since we make a bet of s1 = (s11 || s12) and s2 = (s21 || s22),
        # we actually enumerate over LESS candidates s1 and s2!
        logL[1] = (logS[11] + logS[12]) - logq * r  # logS[1] - logq * r
        logL[2] = (logS[21] + logS[22]) - logq * r  # logS[2] - logq * r
        assert logS[11] + logS[12] <= logS[1]
        assert logS[21] + logS[22] <= logS[2]

        # Analyse the runtime
        log_time_error_guess = RR(((r + 1) // 2) * log(error_width))

        # Note: (1 - 1/q^r)^R1 = (1 + 1/q^r)^{q^r * -R1/q^r} ~ e^{-R1/q^r}.
        prob_rep_dies = exp(-exp(logR[1] - r * logq)) if r*logq > 20 else exp(exp(logR[1]) * log(1 - exp(-r * logq)))
        # Probability that at least one of the representations survives the filter:
        prob_rep_survives = 1.0 - prob_rep_dies
        if prob_rep_survives == 0.0:
            # Use approximation: 1 - e^{-1/x} ~ 1/x
            prob_rep_survives = exp(logR[1] - r * logq)

        # Probability of matching: 0.5^{m-r},
        # for Odlyzko-style matching on s1 and s2 respectively.
        log_time_collisions = [logL[a] + logL[b] - (m - r) * RR(log(2))
                               for (a, b) in [(11, 12), (21, 22)]]
        log_time_lists = logL.values()
        log_runtime = log_time_error_guess + sum_log(*log_time_lists, *log_time_collisions)

        # Analyse the success probability
        bet_s1 = logS[11] + logS[12] - logS[1]  # < 0
        bet_s2 = logS[21] + logS[22] - logS[2]  # < 0

        log_probability = bet_s1 + bet_s2
        repetitions = prob_amplify(target_success_probability,
                                   prob_rep_survives * exp(log_probability))

        cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
        cost.register_impermanent(rop=True, mem=False)
        cost = cost.repeat(repetitions)
        cost["tag"] = "[May21]"
        return cost

    __name__ = "Meet (Rep-0)"


odlyzko = Odlyzko()
meet_rep0 = MeetRep0()
