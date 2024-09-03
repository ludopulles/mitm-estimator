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


def meet_rep_0(omega):
    # Theorem 1 from [May21].

    # Times for L^{(2)} and L^{(1)} respectively.
    return max(0.5 * H2(omega / 4), H2(omega / 4) - omega)


###################################################################################################
# Helper functions for REP-1
###################################################################################################

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
    assert omega_2 <= 0.250  # Otherwise, weird things start to happen.
    x_2 = (1 - 2 * omega_2)
    low, high = 0, 0.00672
    for _ in range(20):  # absolute error < 1e-9
        vareps_3 = (low + high) / 2
        if 2 * omega_2 + x_2 * H2(vareps_3 / x_2) <= 0.5 * H2(omega_2 / 2 + vareps_3):
            low = vareps_3  # Find more representations => increase vareps_3
        else:
            high = vareps_3  # Find less representations => decrease vareps_3
    return (low + high) / 2


def time_omega2(omega_2):
    """
    Evaluates the optimal cost to build all the lists at levels 3 and 4, for depth 4, given
    omega_2, by taking the optimal value for vareps_3 by the above.
    As a function of omega_2, the time is strictly increasing on [0, 0.250]
    """
    return 0.5 * H2(omega_2 / 2 + solve_vareps_3(omega_2))  # T_3 = T_4.


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
    x_1 = (1 - 2 * omega_1)

    low, high = 0, 0.0300
    for _ in range(20):  # absolute error < 1e-8
        vareps_2 = (low + high) / 2
        if 2 * omega_1 + x_1 * H2(vareps_2 / x_1) <= 2.0 * time_omega2(omega_1 / 2 + vareps_2):
            low = vareps_2  # Find more representations => increase vareps_2
        else:
            high = vareps_2  # Find less representations => decrease vareps_2
    return (low + high) / 2


def time_omega1(omega_1):
    """
    Evaluates the optimal cost to build all the lists at levels 3 and 4, for depth 4, given
    omega_2, by taking the optimal value for vareps_3 by the above.
    As a function of omega_1, the time is strictly increasing on [0, 0.250]
    """
    return time_omega2(omega_1 / 2 + solve_vareps_2(omega_1))  # T_3 = T_4.


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
    x_0 = (1 - 2 * omega_0)

    def T_12(vareps_1):
        omega_1 = omega_0 / 2 + vareps_1
        vareps_2 = solve_vareps_2(omega_1)
        omega_2 = omega_1 / 2 + vareps_2

        x_1 = (1 - 2 * omega_1)
        R_1 = 2 * omega_0 + x_0 * H2(vareps_1 / x_0)
        R_2 = 2 * omega_1 + x_1 * H2(vareps_2 / x_1)
        T_1 = 2 * H2(omega_2) - R_1 - R_2  # 2S^{(2)} - R_1 - R_2
        T_2 = .5 * R_2  # 2S^{(3)} - R_2 - R_3
        return T_1, T_2

    # Note: this does not work for high weights, as the function T_1 - T_4 is not strictly
    # increasing as a function of vareps_1.
    low, high = 0, 0.100

    t12_high = T_12(high)
    if t12_high[0] <= t12_high[1]:
        # Hah, try binary search to find vareps_1 where T_1 = T_2.
        for _ in range(20):  # absolute error < 1e-8
            vareps_1 = (low + high) / 2
            T_1, T_2 = T_12(vareps_1)
            if T_1 >= T_2:
                low = vareps_1  # Find more representations => increase vareps_1
            else:
                high = vareps_1  # Find less representations => decrease vareps_1
        return (low + high) / 2
    else:
        # Brute-force w.r.t. vareps_1 to minimize max(T_1, T_2 = T_3 = T_4).
        best = (inf, 0.0)
        for vareps_1 in np.arange(0.001, 0.100, 0.001):
            T_1, T_2 = T_12(vareps_1)
            best = min(best, (max(T_1, T_2), vareps_1))
        return best[1]


def meet_rep_1(omega):
    omega_0 = omega / 2
    vareps_1 = solve_vareps_1(omega_0)
    omega_1 = omega_0 / 2 + vareps_1
    # Warning! Do not assume T_1, T_2, T_3, T_4 are balanced: T_1 might be much slower than T_2.
    # return time_omega1(omega_1)
    vareps_2 = solve_vareps_2(omega_1)
    omega_2 = omega_1 / 2 + vareps_2

    x_0, x_1 = (1 - 2 * omega_0), (1 - 2 * omega_1)
    R_1 = 2 * omega_0 + x_0 * H2(vareps_1 / x_0)
    R_2 = 2 * omega_1 + x_1 * H2(vareps_2 / x_1)

    T_1 = 2 * H2(omega_2) - R_1 - R_2  # 2S^{(2)} - R_1 - R_2
    T_2 = .5 * R_2  # 2S^{(3)} - R_2 - R_3
    return max(T_1, T_2)


def test_rep_1(omega, vareps_1, vareps_2, vareps_3):
    # Determine relative weights
    omega_0 = omega / 2
    omega_1 = omega_0 / 2 + vareps_1
    omega_2 = omega_1 / 2 + vareps_2
    omega_3 = omega_2 / 2 + vareps_3

    # Used for R_1.
    x_0, x_1, x_2 = 1 - 2 * omega_0,  1 - 2 * omega_1, 1 - 2 * omega_2

    # S_1 = H2(omega_1)
    S_2, S_3 = H2(omega_2), H2(omega_3)

    R_1 = 2 * omega_0 + x_0 * H2(vareps_1 / x_0)
    R_2 = 2 * omega_1 + x_1 * H2(vareps_2 / x_1)
    R_3 = 2 * omega_2 + x_2 * H2(vareps_3 / x_2)

    # L_1 = S_1 - R_1
    L_2, L_3, L_4 = S_2 - R_2, S_3 - R_3, S_3 / 2
    assert 2*L_3 - (R_2 - R_3) >= L_2

    # Assuming q > n >> 2, T_0 becomes negligible.
    # T_0 = 2 * L_1 - (1.j0 - R_1 / log2(q))
    T_1, T_2, T_3, T_4 = 2 * L_2 - R_1 + R_2, 2 * L_3 - R_2 + R_3, 2 * L_4 - R_3, L_4

    # Note: for optimal parameters, try to have
    # T_1 ~ T_2 ~ T_3 ~ T_4.
    # Given omega, this gives equations for vareps_{1,2,3} to satisfy.

    # print(f'T_{{1,2,3,4}} = {T_1:8.6f}, {T_2:8.6f}, {T_3:8.6f}, {T_4:8.6f}')
    return max(T_1, T_2, T_3, T_4)


def ceil_print(x, decimals):
    x = ceil(10**decimals * x) / 10**decimals
    return ('{:.' + str(decimals) + 'f}').format(x)


def __main__():
    table_5 = [
        (0.300, 0.591, 0.469, 0.298),
        (0.375, 0.665, 0.523, 0.323),
        (0.441, 0.716, 0.561, 0.340),
        (0.500, 0.750, 0.588, 0.356),
        (0.620, 0.790, 0.625, 0.389),
        (0.667, 0.793, 0.634, 0.407),
    ]

    print('omega | Odlyzko | [May21] | REP-0 | [May21] | REP-1 | [May21] |')
    print('------+---------+---------+-------+---------+-------+---------+')
    for omega, May_odly, May_REP0, May_REP1 in table_5:
        odly = odlyzko(omega)
        REP0 = meet_rep_0(omega)
        REP1 = meet_rep_1(omega)
        print(f'{omega:5.3f} |   {odly:5.3f} | {May_odly:5.3f}   '
              f'| {REP0:5.3f} | {May_REP0:5.3f}   '
              f'| {REP1:5.3f} | {May_REP1:5.3f}   |')

    print("\nCompiling plot...")

    xs = list(np.arange(0.01, 2/3, 0.01))
    plt.plot(xs, [odlyzko(x) / lgS(x) for x in xs], label='Odlyzko')
    plt.plot(xs, [howard_graham(x) / lgS(x) for x in xs], label='Howard-Graham')
    plt.plot(xs, [meet_rep_0(x) / lgS(x) for x in xs], label='REP-0')
    plt.plot(xs, [meet_rep_1(x) / lgS(x) for x in xs], label='REP-1')

    plt.xlabel('omega = lim_{n -> oo} w/n')
    plt.ylabel('log(time) / log(search-space)')
    plt.legend()
    plt.show()

    # xs = list(np.arange(0.001, 0.5, 0.001))
    # plt.plot(xs, [H2(x) for x in xs], label='H2(x)')
    # plt.plot(xs, [8*x for x in xs], label='y = 8x')
    # plt.plot(xs, [-2*x*log2(x) for x in xs], label='y = x lg(x)')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    __main__()
