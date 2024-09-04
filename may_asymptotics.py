from functools import partial
from math import ceil, inf, log2

import matplotlib.pyplot as plt
import numpy as np


def H(*xs):
    y = 1.0 - sum(xs)
    assert y > 0
    return -sum(x * log2(x) for x in xs) - y * log2(y)


def H2(x):
    """
    Return H(x, x, 1 - 2x).
    """
    assert 0 < x < 0.5
    return -2 * x * log2(x) - (1.0 - 2 * x) * log2(1.0 - 2 * x)


def lgS(omega):
    """
    Return |S| = #{x \\in T^n(omega)}.
    """
    return H2(omega / 2)
    # return H(omega / 2, omega / 2)


def odlyzko(omega):
    """
    :param omega: relative weight (omega = w / n)
    Complexity: ~ sqrt(|S|).
    """
    return lgS(omega) / 2


def howard_graham(omega):
    return H2(omega / 4) - omega / 2


def meet_rep0_depth2(omega):
    # Theorem 1 from [May21].
    # Times for L^{(2)} and L^{(1)} respectively.
    return max(0.5 * H2(omega / 4), H2(omega / 4) - omega)


def meet_rep0_depth3(omega):
    S_2 = H2(omega / 8)

    T_1 = 2 * S_2 - 1.50 * omega
    T_2 = 1 * S_2 - 0.5 * omega
    T_3 = 0.5 * S_2
    assert T_1 > H2(omega / 4) - omega
    return max(T_1, T_2, T_3)


def meet_rep0_depth4(omega):
    S_2, S_3 = H2(omega / 8), H2(omega / 16)

    T_1 = 2 * S_2 - 1.50 * omega
    T_2 = 2 * S_3 - 0.75 * omega
    T_3 = 1 * S_3 - 0.25 * omega
    T_4 = 0.5 * S_3
    assert T_1 > H2(omega / 4) - omega
    assert T_2 > H2(omega / 8) - omega / 2
    return max(T_1, T_2, T_3, T_4)


###################################################################################################
# Helper functions for REP-1
###################################################################################################
BS_PRECISION = 20


def binary_search_f(f, left, right):
    """
    Given a continuous increasing function f such that f(left) < 0 and f(right) > 0, return `x`
    such that f(x) = 0, using `BS_PRECISION` iterations.
    """
    for _ in range(BS_PRECISION):  # absolute error < 1e-9
        mid = 0.5 * (left + right)
        if f(mid) < 0:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def ternary_search_f(f, left, right):
    """
    Given a continuous function f such that there is a local minimum `x` in [left, right], return
    `x`.
    """
    for _ in range(BS_PRECISION):  # absolute error < 1e-9
        a, b = (2 * left + right) / 3, (left + 2 * right) / 3
        if f(a) < f(b):
            right = b
        else:
            left = a
    return 0.5 * (left + right)


def solve_vareps_3(omega_2):
    """
    Inside REP-1, apply binary search to solve the following equation with respect to vareps_3:
        R^{(3)} = 0.5 S^{(3)}
    This translates to:
        2omega_2 + (1-2omega_2) H_2(vareps_3 / (1-2omega_2)) = 0.5 H_2(omega_2/2 + vareps_3)

    As a function of omega_2:
    - omega_2 in [0, 0.105] => vareps_3 is increasing,
    - omega_2 ~ 0.105 => vareps_3 is maximal: 0.00671958,
    - omega_2 in [0.105, 0.250] => vareps_3 is decreasing.
    """
    if H2(omega_2 / 2) - 2 * omega_2 < 0.5 * H2(omega_2 / 2):
        # Basically REP-0, because vareps_3 = 0 is optimal.
        return 0.0
    x_2 = (1 - 2 * omega_2)

    def f(x):
        return 2 * omega_2 + x_2 * H2(x / x_2) - 0.5 * H2(omega_2 / 2 + x)
    return binary_search_f(f, 0, min(x_2, 0.00672))


def time_omega2(omega_2):
    """
    Evaluates the optimal cost to build all the lists at levels 3 and 4, for depth 4, given
    omega_2, by taking the optimal value for vareps_3 by the above.
    As a function of omega_2, the time is strictly increasing on [0, 0.250]
    """
    return 0.5 * H2(omega_2 / 2 + solve_vareps_3(omega_2))  # T_3 <= T_4.


def time_depth3(omega_1, eps_2):
    """
    Return the needed time for level 1 (using depth=3) by picking eps_3 optimally.
    :param omega_1: a target weight for the layer below,
    :param eps_2: the increase of weight for the current layer
    """
    omega_2 = omega_1 / 2 + eps_2
    eps_3 = solve_vareps_3(omega_2)
    omega_3 = omega_2 / 2 + eps_3

    S_3 = H2(omega_3)
    R_2 = 2 * omega_1 + (0 if eps_2 == 0 else (1 - 2 * omega_1) * H2(eps_2 / (1 - 2 * omega_1)))
    R_3 = 2 * omega_2 + (0 if eps_3 == 0 else (1 - 2 * omega_2) * H2(eps_3 / (1 - 2 * omega_2)))
    return 2 * S_3 - R_2 - R_3


def solve_vareps_2(omega_1):
    """
    Inside REP-1, apply binary search to solve the following equation with respect to vareps_2:
        R^{(2)} = S^{(3)} (= 2 T_3 = 2 T_4).
    This translates to:
        2omega_1 + (1-2omega_1) H_2(vareps_2 / (1-2omega_1))
        = H_2(omega_1/4 + vareps_2/2 + vareps_3),
    where vareps_3 is chosen optimally to satisfy the equation from `solve_vareps_3`.
    Note: T_3 = T_4 both increase strictly as function of vareps_2, but T_2 ~ -R_2 is decreasing as
    a function of vareps_2. Hence, we can binary search on `vareps_2`.

    As a function of omega_1:
    - omega_1 in [0, 0.214] => vareps_2 is increasing,
    - omega_1 ~ 0.214 => vareps_2 is maximal: 0.02877658,
    - omega_1 in [0.214, 0.392] => vareps_2 is decreasing.
    """
    def f(eps_2):
        return time_omega2(omega_1 / 2 + eps_2) - time_depth3(omega_1, eps_2)

    opt_eps_2 = ternary_search_f(partial(time_depth3, omega_1), 0, 0.5 - omega_1)
    if f(opt_eps_2) <= 0:
        return opt_eps_2
    else:
        return binary_search_f(f, 0, opt_eps_2)


def time_omega1(omega_1):
    """
    Evaluates the optimal cost to build all the lists at levels 3 and 4, for depth 4, given
    omega_2, by taking the optimal value for vareps_3 by the above.
    As a function of omega_1, the time is strictly increasing on [0, 0.250]
    """
    return time_omega2(omega_1 / 2 + solve_vareps_2(omega_1))  # T_3 = T_4.


def time_depth4(omega_0, eps_1):
    """
    Return the needed time for level 1 (using depth=3) by picking eps_3 optimally.
    :param omega_1: a target weight for the layer below,
    :param eps_2: the increase of weight for the current layer
    """
    omega_1 = omega_0 / 2 + eps_1
    eps_2 = solve_vareps_2(omega_1)
    omega_2 = omega_1 / 2 + eps_2

    S_2 = H2(omega_2)
    R_1 = 2 * omega_0 + (0 if eps_1 == 0 else (1 - 2 * omega_0) * H2(eps_1 / (1 - 2 * omega_0)))
    R_2 = 2 * omega_1 + (0 if eps_2 == 0 else (1 - 2 * omega_1) * H2(eps_2 / (1 - 2 * omega_1)))
    return 2 * S_2 - R_1 - R_2


def solve_vareps_1(omega_0):
    """
    Inside REP-1, apply binary search to solve the following equation with respect to vareps_1:
        2S^{(2)} - R^{(1)} = 1.5 S^{(3)}.
    This translates to:
        2H_2(omega_2) - 2omega_0 - (1-2omega_0) H_2(vareps_1 / (1-2omega_0))
        = 1.5 H_2(omega_3).

    where vareps_2, vareps_3 are chosen optimally to satisfy the equation from `solve_vareps_2` and
    `solve_vareps_3`.
    """
    def f(eps_1):
        return time_omega1(omega_0 / 2 + eps_1) - time_depth4(omega_0, eps_1)

    opt_eps_1 = ternary_search_f(partial(time_depth4, omega_0), 0, 0.5 - omega_0)
    if f(opt_eps_1) <= 0:
        return opt_eps_1
    else:
        return binary_search_f(f, 0, opt_eps_1)

    x_0 = (1 - 2 * omega_0)

    def T_12(vareps_1):
        omega_1 = omega_0 / 2 + vareps_1
        vareps_2 = solve_vareps_2(omega_1)
        omega_2 = omega_1 / 2 + vareps_2

        x_1 = (1 - 2 * omega_1)
        R_1 = 2 * omega_0 + (0 if vareps_1 == 0 else x_0 * H2(vareps_1 / x_0))
        R_2 = 2 * omega_1 + x_1 * H2(vareps_2 / x_1)
        T_1 = 2 * H2(omega_2) - R_1 - R_2  # 2S^{(2)} - R_1 - R_2
        T_2 = .5 * R_2  # 2S^{(3)} - R_2 - R_3
        return T_1, T_2

    # vareps_1s = np.arange(1e-3, 0.1, 1e-3)
    # T1, T2 = zip(*[T_12(x) for x in vareps_1s])
    # plt.plot(vareps_1s, T1, label='T_1')
    # plt.plot(vareps_1s, T2, label='T_2')
    # # plt.plot(vareps_1s, [max(T_12(x)) for x in vareps_1s], label='max(T)')
    # plt.title(f'omega_0 = {omega_0}')
    # plt.xlabel('vareps_1')
    # plt.show()

    # Note: this does not work for high weights, as the function T_1 - T_4 is not strictly
    # increasing as a function of vareps_1.
    low, high = 0, 0.100

    T_1, T_2 = T_12(high)
    if T_1 <= T_2:
        # Hah, try binary search to find vareps_1 where T_1 = T_2.
        for _ in range(BS_PRECISION):  # absolute error < 1e-8
            vareps_1 = (low + high) / 2
            T_1, T_2 = T_12(vareps_1)
            if T_1 >= T_2:
                low = vareps_1  # Find more representations => increase vareps_1
            else:
                high = vareps_1  # Find less representations => decrease vareps_1

        # The interval [0, low] is dominated by the runtime of T_1. Try to minimize that one.
        # Especially for low weights (omega <= 0.0252, omega_0 <= 0.0176) it appears that T_1 is an
        # INCREASING function as function of vareps_1 (and vareps_2, vareps_3), while for larger
        # weights, it is DECREASING.
        vareps_1 = (low + high) / 2
        time = max(T_12(vareps_1))
        if max(T_12(0.0)) < time:
            return 0.0

        while True:
            alt_time = max(T_12(vareps_1 / 2))
            if alt_time < time:
                vareps_1, time = vareps_1 / 2, alt_time
            else:
                break
        return vareps_1
    else:
        # Brute-force w.r.t. vareps_1 to minimize max(T_1, T_2 = T_3 = T_4).
        best = (inf, 0.0)
        for vareps_1 in np.arange(0.001, 0.1, 0.001):
            T_1, T_2 = T_12(vareps_1)
            best = min(best, (max(T_1, T_2), vareps_1))
        return best[1]


def meet_rep1_depth2(omega):
    """
    This is a kind of cheat, because we use the terminolgy of depth=4, but shifted by two.
    That is, we only need to solve for vareps_3 which is (actually) the added weights at level 1.
    """
    omega_2 = omega / 2
    vareps_3 = solve_vareps_3(omega_2)
    omega_3 = omega_2 / 2 + vareps_3

    x_2 = 1 - omega
    S_3 = H2(omega_3)
    R_3 = omega + (0 if vareps_3 == 0 else x_2 * H2(vareps_3 / x_2))

    # Assuming q > n >> 2, T_0 becomes negligible.
    # T_0 = 2 * L_1 - (1 - R_1 / log2(q))
    T_3, T_4 = S_3 - R_3, S_3 / 2
    return max(T_3, T_4)


def meet_rep1_depth3(omega):
    """
    This is a kind of cheat, because we use the terminolgy of depth=4, but shifted by one.
    That is, we only need to solve for vareps_2, vareps_3 which are (actually) the added weights at
    level 1 and 2 respectively.
    """
    omega_1 = omega / 2
    vareps_2 = solve_vareps_2(omega_1)
    omega_2 = omega_1 / 2 + vareps_2
    vareps_3 = solve_vareps_3(omega_2)
    omega_3 = omega_2 / 2 + vareps_3

    # Used for R_1.
    x_1, x_2 = 1 - 2 * omega_1, 1 - 2 * omega_2
    S_3 = H2(omega_3)

    R_2 = 2 * omega_1 + (0 if vareps_2 == 0 else x_1 * H2(vareps_2 / x_1))
    R_3 = 2 * omega_2 + (0 if vareps_3 == 0 else x_2 * H2(vareps_3 / x_2))

    # L_1 = S_1 - R_1
    L_3, L_4 = S_3 - R_3, S_3 / 2

    # Assuming q > n >> 2, T_0 becomes negligible.
    # T_0 = 2 * L_1 - (1 - R_1 / log2(q))
    T_2, T_3, T_4 = 2 * L_3 - R_2 + R_3, 2 * L_4 - R_3, L_4
    return max(T_2, T_3, T_4)


def meet_rep1_depth4(omega):
    omega_0 = omega / 2
    vareps_1 = solve_vareps_1(omega_0)
    omega_1 = omega_0 / 2 + vareps_1
    # Warning! Do not assume T_1, T_2, T_3, T_4 are balanced: T_1 might be much slower than T_2.
    # return time_omega1(omega_1)
    vareps_2 = solve_vareps_2(omega_1)
    omega_2 = omega_1 / 2 + vareps_2

    x_0, x_1 = (1 - 2 * omega_0), (1 - 2 * omega_1)
    R_1 = 2 * omega_0 + (0 if vareps_1 == 0 else x_0 * H2(vareps_1 / x_0))
    R_2 = 2 * omega_1 + x_1 * H2(vareps_2 / x_1)

    T_1 = 2 * H2(omega_2) - R_1 - R_2  # 2S^{(2)} - R_1 - R_2
    T_2 = .5 * R_2  # 2S^{(3)} - R_2 - R_3
    return max(T_1, T_2)


def time_rep1(omega, vareps_1, vareps_2, vareps_3, all_times=False):
    # Determine relative weights
    omega_0 = omega / 2
    omega_1 = omega_0 / 2 + vareps_1
    omega_2 = omega_1 / 2 + vareps_2
    omega_3 = omega_2 / 2 + vareps_3

    # Used for R_1.
    x_0, x_1, x_2 = 1 - 2 * omega_0,  1 - 2 * omega_1, 1 - 2 * omega_2

    # S_1 = H2(omega_1)
    S_2, S_3 = H2(omega_2), H2(omega_3)

    R_1 = 2 * omega_0 + (0 if vareps_1 == 0 else x_0 * H2(vareps_1 / x_0))
    R_2 = 2 * omega_1 + (0 if vareps_2 == 0 else x_1 * H2(vareps_2 / x_1))
    R_3 = 2 * omega_2 + (0 if vareps_3 == 0 else x_2 * H2(vareps_3 / x_2))

    # L_1 = S_1 - R_1
    L_2, L_3, L_4 = S_2 - R_2, S_3 - R_3, S_3 / 2
    assert 2*L_3 - (R_2 - R_3) >= L_2

    # Assuming q > n >> 2, T_0 becomes negligible.
    # T_0 = 2 * L_1 - (1 - R_1 / log2(q))
    T_1, T_2, T_3, T_4 = 2 * L_2 - R_1 + R_2, 2 * L_3 - R_2 + R_3, 2 * L_4 - R_3, L_4

    # Note: for optimal parameters, try to have
    # T_1 ~ T_2 ~ T_3 ~ T_4.
    # Given omega, this gives equations for vareps_{1,2,3} to satisfy.
    # print(f'T_{{1,2,3,4}} = {T_1:8.6f}, {T_2:8.6f}, {T_3:8.6f}, {T_4:8.6f}')

    if all_times:
        return T_1, T_2, T_3, T_4
    else:
        return max(T_1, T_2, T_3, T_4)


def ceil_print(x, decimals):
    x = ceil(10**decimals * x) / 10**decimals
    return ('{:.' + str(decimals) + 'f}').format(x)


###############################################################################
# Testing functions
###############################################################################
def verify_table_5():
    table_5 = [
        (0.300, 0.591, 0.469, 0.298),
        (0.375, 0.665, 0.523, 0.323),
        (0.441, 0.716, 0.561, 0.340),
        (0.500, 0.750, 0.588, 0.356),
        (0.620, 0.790, 0.625, 0.389),
        (0.667, 0.793, 0.634, 0.407),
    ]

    print('omega | Odlyzko       | REP-0                         | REP-1                         |')
    print('omega | ours  | May21 |  d=2  |  d=3  |  d=4  | May21 |  d=2  |  d=3  |  d=4  | May21 |')
    print('------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+')
    for omega, May_odly, May_REP0, May_REP1 in table_5:
        odly = odlyzko(omega)

        REP0d2 = meet_rep0_depth2(omega)
        REP0d3 = meet_rep0_depth3(omega)
        REP0d4 = meet_rep0_depth4(omega)

        REP1d2 = meet_rep1_depth2(omega)
        REP1d3 = meet_rep1_depth3(omega)
        REP1d4 = meet_rep1_depth4(omega)

        print(f'{omega:5.3f} | {odly:5.3f} | {May_odly:5.3f} '
              f'| {REP0d2:5.3f} | {REP0d3:5.3f} | {REP0d4:5.3f} | {May_REP0:5.3f} '
              f'| {REP1d2:5.3f} | {REP1d3:5.3f} | {REP1d4:5.3f} | {May_REP1:5.3f} |')

    xs = list(np.arange(0.01, 1, 0.02))

    # plt.plot(xs, [1 for x in xs], label='Brute-force')
    plt.plot(xs, [odlyzko(x) / lgS(x) for x in xs], label='Odlyzko')
    plt.plot(xs, [howard_graham(x) / lgS(x) for x in xs], label='Howard-Graham')

    # REP-0
    plt.plot(xs, [meet_rep0_depth2(x) / lgS(x) for x in xs], label='REP-0 (depth 2)')
    plt.plot(xs, [meet_rep0_depth3(x) / lgS(x) for x in xs], label='REP-0 (depth 3)')
    plt.plot(xs, [meet_rep0_depth4(x) / lgS(x) for x in xs], label='REP-0 (depth 4)')

    # REP-1
    plt.plot(xs, [meet_rep1_depth2(x) / lgS(x) for x in xs], label='REP-1 (depth 2)')
    plt.plot(xs, [meet_rep1_depth3(x) / lgS(x) for x in xs], label='REP-1 (depth 3)')
    plt.plot(xs, [meet_rep1_depth4(x) / lgS(x) for x in xs], label='REP-1 (depth 4)')

    plt.xlabel('omega ~ w/n')
    plt.ylabel('log(time) / log(search-space)')
    plt.legend()
    plt.show()


def analyze_depth2():
    # The conclusion: for omega <= 0.58 (cf. Remark 1 [May21]), note that most time is spent on
    # building L^{(1)}. Thus, there is no use making more levels, since this will give wrong
    # s_{1,2}^(1)'s which need to be filtered out, making the algorithm slower. Thus, depth 2 is
    # optimal for REP-0 when omega <= 0.58.

    # Just when vareps = 0:
    # xs = np.arange(0.001, 1.0, 0.001)
    # plt.plot(xs, [0.5 * H2(x/4) for x in xs], label='Time for L^{(2)}')
    # plt.plot(xs, [H2(x / 4) - x for x in xs], label='Time for L^{(1)}')
    # plt.xlabel('omega')
    # plt.legend()
    # plt.show()

    # Shows three different regimes with respect to time spent in level 1 and in level 2.
    for omega in [0.3, 0.58, 0.8]:
        omega_2 = omega / 2
        x_2 = 1 - 2 * omega_2

        xs, Ts = np.arange(0, x_2/2, 0.001), []
        for vareps_3 in xs:
            S_3 = H2(omega_2 / 2 + vareps_3)
            R_3 = 2 * omega_2 + (0 if vareps_3 == 0 else x_2 * H2(vareps_3 / x_2))
            Ts.append((S_3 - R_3, S_3 / 2))
        Ts = list(zip(*Ts))
        plt.plot(xs, Ts[0], label='T_1')
        plt.plot(xs, Ts[1], label='T_2')
        plt.title(f'omega = {omega}')
        plt.legend()
        plt.show()


def analyze_depth3():
    for omega in [0.6, 0.86, 0.95]:
        omega_1 = omega / 2

        x_1 = 1 - 2 * omega_1
        xs, Ts = np.arange(0, x_1/2, 0.001), []
        for vareps_2 in xs:
            omega_2 = omega_1 / 2 + vareps_2
            x_2 = 1 - 2 * omega_2
            vareps_3 = solve_vareps_3(omega_2)
            omega_3 = omega_2 / 2 + vareps_3

            S_3 = H2(omega_3)
            R_2 = 2 * omega_1 + (0 if vareps_2 == 0 else x_1 * H2(vareps_2 / x_1))
            R_3 = 2 * omega_2 + (0 if vareps_3 == 0 else x_2 * H2(vareps_3 / x_2))

            L_4 = R_3 = S_3 / 2
            L_3, L_4 = S_3 - R_3, S_3 / 2
            Ts.append((2 * L_3 - R_2 + R_3, 2 * L_4 - R_3, L_4))
        Ts = list(zip(*Ts))
        plt.plot(xs, Ts[0], label='T_1')
        plt.plot(xs, Ts[1], label='T_2 = T_3')
        plt.legend()
        plt.show()


def analyze_depth4():
    for omega in [0.5, 0.77, 0.85]:
        omega_0 = omega / 2
        x_0 = 1 - 2 * omega_0
        xs, Ts = np.arange(0, x_0/2, 0.001), []
        for eps_1 in xs:
            omega_1 = omega_0 / 2 + eps_1
            x_1 = 1 - 2 * omega_1
            eps_2 = solve_vareps_2(omega_1)
            omega_2 = omega_1 / 2 + eps_2
            eps_3 = solve_vareps_3(omega_2)
            omega_3 = omega_2 / 2 + eps_3

            x_1, x_2 = 1 - 2 * omega_1, 1 - 2 * omega_2

            S_2 = H2(omega_2)
            S_3 = H2(omega_3)

            R_1 = 2 * omega_0 + (0 if eps_1 == 0 else x_0 * H2(eps_1 / x_0))
            R_2 = 2 * omega_1 + (0 if eps_2 == 0 else x_1 * H2(eps_2 / x_1))
            R_3 = 2 * omega_2 + (0 if eps_3 == 0 else x_2 * H2(eps_3 / x_2))

            # L_1 = S_1 - R_1
            L_2, L_3, L_4 = S_2 - R_2, S_3 - R_3, S_3 / 2

            T_1, T_2, T_3, T_4 = 2 * L_2 - R_1 + R_2, 2 * L_3 - R_2 + R_3, 2 * L_4 - R_3, L_4
            Ts.append((T_1, T_2, T_3, T_4))
        Ts = list(zip(*Ts))
        plt.plot(xs, Ts[0], label='T_1')
        plt.plot(xs, Ts[1], label='T_2')
        plt.plot(xs, Ts[2], label='T_3')
        plt.plot(xs, Ts[3], label='T_4')
        plt.legend()
        plt.show()


def __main__():
    # analyze_depth3()
    # analyze_depth4()
    verify_table_5()
    return

    # for omega in np.arange(0.0001, 0.01, 0.0001):
    for omega in [0.001]:
        omega_0 = omega / 2
        vareps_1 = solve_vareps_1(omega_0)
        omega_1 = omega_0 / 2 + vareps_1
        vareps_2 = solve_vareps_2(omega_1)
        omega_2 = omega_1 / 2 + vareps_2
        vareps_3 = solve_vareps_3(omega_2)

        T_1, T_2, T_3, T_4 = time_rep1(omega_0, vareps_1, vareps_2, vareps_3, True)
        print()
        print('Carte blanche:')
        print(f'omega {omega:8.6f}: {vareps_1:8.6f}, {vareps_2:8.6f}, {vareps_3:8.6f} '
              f'-> {T_1:7.5f}, {T_2:7.5f}, {T_3:7.5f}, {T_4:7.5f}')
        print()

        if vareps_1 == 0.0:
            for x in np.arange(0, 0.01, 0.001):
                omega_2 = omega_1 / 2 + x
                vareps_3 = solve_vareps_3(omega_2)
                T_1, T_2, T_3, T_4 = time_rep1(omega_0, 0.0, x, vareps_3, True)
                print(f'x={x:6.4f}: eps_3={vareps_3:6.4f} '
                      f'-> {T_1:7.5f}, {T_2:7.5f}, {T_3:7.5f}, {T_4:7.5f}')


if __name__ == "__main__":
    __main__()
