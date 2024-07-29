# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using primal attacks.

See :ref:`LWE Primal Attacks` for an introduction what is available.

"""
from sage.all import binomial, exp, floor, log, RR

from .cost import Cost
from .lwe_parameters import LWEParameters
from .prob import amplify as prob_amplify


class CombinatorialMeet:
    """
    Estimate cost of solving LWE via MeetLWE [May21]_, using the Rep-0 representation techniques.
    """
    def log_comb(n, *L):
        """
        Returns log(n! / (L[0]! L[1]! ... L[-1]!)), where sum(L) <= n.
        This is the number of ways to arrange `n` balls s.t. L[i] balls have colour i (i=0,1,...).
        EXAMPLE::

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

    def split_up(n):
        """
        Split up `n` evenly is: one equals `ceil(n / 2)`, and the other `floor(n / 2)`.
        :return: pair (a, b) such that a + b = n.
        """
        return (n + 1) // 2, n // 2

    def __call__(
        self,
        params: LWEParameters,
        log_level=1,
        **kwds,
    ):
        """
        Estimate cost of solving LWE via Meet-LWE [May21]_.

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
        :return: A cost dictionary.

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``mem``: The number of entries in a complete table.

        EXAMPLE::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=200, q=127, Xs=ND.SparseTernary(200, 10), Xe=ND.UniformMod(200, 10))
            >>> LWE.primal_usvp(params)
            rop: ≈2^87.6, tag: meet-lwe

        """
        log_comb = CombinatorialMeet.log_comb
        sum_log  = CombinatorialMeet.sum_log
        split_up = CombinatorialMeet.split_up

        # Note: this gives issues for the "GLP I" parameter set, so don't normalize.
        # params = LWEParameters.normalize(params)

        print(f'Secret distribution: {repr(params.Xs)}, tag = {params.Xs.tag}')

        # Check for ternary instead of sparse ternary.
        assert params.Xs.tag == "SparseTernary", "Secret distribution has to be ternary."
        assert params.Xe.is_bounded, "Error distribution has to be bounded."
        assert params.Xs.mean == 0, "Expected #1's == #-1's."

        n, m, logq = params.n, params.m, RR(log(params.q))
        error_width = params.Xe.bounds[1] - params.Xe.bounds[0] + 1

        logS = {}  # log(size of search space)
        logR = {}  # log(number of representatives)
        logL = {}  # log(size of list of stored candidates)

        # This is *half* the hamming weight of `s`.

        weight_s = params.Xs.get_hamming_weight(n) // 2
        # Note: out of the `n` secret coefficients, `weight_s` are +1 and `weight_s` are -1.

        weight_s1, weight_s2 = split_up(weight_s)

        weight_s11, weight_s12 = split_up(weight_s1)
        weight_s21, weight_s22 = split_up(weight_s2)
        n_left, n_right = split_up(n)

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
        r = floor(logR[1] / logq)  # floor( log_q(R(1)) ) = floor( log(R(1)) / log(q) ).

        # Warning: since we make a bet of s1 = (s11 || s12) and s2 = (s21 || s22), we actually
        # enumerate over LESS candidates s1 and s2!
        # logL[1] = logS[1] - logq * r
        # logL[2] = logS[2] - logq * r
        logL[1] = (logS[11] + logS[12]) - logq * r
        logL[2] = (logS[21] + logS[22]) - logq * r

        assert logS[11] + logS[12] <= logS[1]

        # Analyse the runtime
        log_time_error_guess = RR(((r + 1) // 2) * log(error_width))

        # Probability of matching: 0.5^{m-r}.
        log_time_collisions = (logL[11] + logL[12] - (m - r) * RR(log(2))  # Odlyzko matching on s1
                               + logL[21] + logL[22] - (m - r) * RR(log(2)))  # Odlyzko matching on s2
        log_time_lists = sum_log(*logL.values())
        log_runtime = log_time_error_guess + sum_log(log_time_lists, log_time_collisions)

        # Analyse the success probability
        bet_s1 = logS[11] + logS[12] - logS[1]  # < 0
        bet_s2 = logS[21] + logS[22] - logS[2]  # < 0

        log_probability = bet_s1 + bet_s2
        repetitions = prob_amplify(0.99, exp(log_probability))

        cost = Cost(rop=exp(log_runtime), mem=exp(log_runtime))
        cost.register_impermanent(rop=True, mem=False)
        return cost.repeat(repetitions)

    __name__ = "combinatorial_meet"


combinatorial_meet = CombinatorialMeet()
