from functools import cache
from math import ceil, log2
import sys

import matplotlib.pyplot as plt
import numpy as np


@cache
def H2(x):
    """
    Return `H(x, x, 1 - 2x)`, where H is the binary entropy function.
    """
    assert 0 < x < 0.5
    return -2 * x * log2(x) - (1.0 - 2 * x) * log2(1.0 - 2 * x)


def lgS(omega):
    """
    Return |S| = #{x \\in T^n(omega)}.
    """
    return H2(omega / 2)


def odlyzko(omega):
    """
    Complexity: ~ sqrt(|S|).
    :param omega: relative weight (omega = w / n)
    """
    return lgS(omega) / 2


def howgrave_graham(omega):
    return H2(omega / 4) - omega / 2


def meet_rep0(depth, omega):
    """
    Asymptotic runtime cost when using REP-0 from [May21] with proportional weight `omega` and a
    tree with depth `depth`.
    Note: any depth > 3 is suboptimal!
    """
    if depth == 2:
        # Theorem 1 in [May21].
        return max(0.5 * H2(omega / 4), H2(omega / 4) - omega)
    else:
        return max(2 * H2(omega / 8) - 1.5 * omega, meet_rep0(depth - 1, omega / 2))


###################################################################################################
# Helper functions for REP-1
###################################################################################################
BS_PRECISION = 20


def binary_search_f(f, left, right):
    """
    Given a continuous increasing function f such that f(left) < 0 and f(right) > 0, return `x`
    such that f(x) = 0, using `BS_PRECISION` iterations.
    Absolute error: (right-left) / 2^{BS_PRECISION}.
    """
    for _ in range(BS_PRECISION):
        mid = 0.5 * (left + right)
        if f(mid) < 0:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def ternary_search_f(f, left, right):
    """
    Given a continuous function f such that there is a local minimum `x` in [left, right], return
    `x`, using `BS_PRECISION` iterations.
    Absolute error: (right-left) * (2/3)^{BS_PRECISION}.
    """
    for _ in range(BS_PRECISION):
        a, b = (2 * left + right) / 3, (left + 2 * right) / 3
        if f(a) < f(b):
            right = b
        else:
            left = a
    return 0.5 * (left + right)


@cache
def optimal_epsilon(depth, omega):
    """
    Tries to find the optimal parameter `epsilon`, which says that at level 1 of a tree of depth
    `depth`, you should have secrets with `omega/2 + eps` of them being 1's (and number of -1's),
    that combine into secrets of weight `omega` at level 0.

    :return: `epsilon`
    """
    def g(eps):
        base_time, sub_time = optimal_times(depth, omega, eps)
        return sub_time - base_time

    if depth == 2:
        # Base case: try to balance the cost of generating the two lists and the merged list.
        if g(0.0) > 0:
            # More time is already spent on the Odlyzko layer, so this is basically REP-0 (eps=0).
            return 0.0
        return binary_search_f(g, 0, 0.5 - omega)

    # First minimize the time spent on the base layer, by performing a ternary search.
    def f(eps):
        return optimal_times(depth, omega, eps)[0]

    opt_eps = ternary_search_f(f, 0, 0.5 - omega)
    if g(opt_eps) > 0:
        # Now, potentially find the sweet spot such that T_{base-layer} = T_{higher-layers},
        # where the maximum is equal to the two.
        return binary_search_f(g, 0, opt_eps)
    return opt_eps


@cache
def optimal_times(depth, omega, eps):
    """
    Return the needed time for level 1, and deeper layers, by picking `delta` optimally.
    :param omega: a target weight for the layer below,
    :param eps: the increase of weight for the current layer.
    :param depth: depth of the tree
    :return: pair of 1) time for level 1 and 2) time for the lower layers
    """
    sub_omega = omega / 2 + eps
    R_1 = 2 * omega + (0 if eps == 0 else (1 - 2 * omega) * H2(eps / (1 - 2 * omega)))
    if depth == 2:
        # Base case.
        S_1 = H2(sub_omega)
        return S_1 - R_1, 0.5 * S_1

    delta = optimal_epsilon(depth - 1, sub_omega)
    sub_time = max(optimal_times(depth - 1, sub_omega, delta))

    S_2 = H2(sub_omega / 2 + delta)
    R_2 = 2 * sub_omega + (0 if delta == 0 else (1 - 2 * sub_omega) * H2(delta / (1 - 2 * sub_omega)))
    return 2 * S_2 - R_1 - R_2, sub_time


def meet_rep1(depth, omega):
    eps = optimal_epsilon(depth, omega / 2)
    return max(optimal_times(depth, omega / 2, eps))


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

        REP0d2 = meet_rep0(2, omega)
        REP0d3 = meet_rep0(3, omega)
        REP0d4 = meet_rep0(4, omega)

        REP1d2 = meet_rep1(2, omega)
        REP1d3 = meet_rep1(3, omega)
        REP1d4 = meet_rep1(4, omega)

        print(f'{omega:5.3f} | {ceil_print(odly, 3)} | {May_odly:5.3f} '
              f'| {ceil_print(REP0d2, 3)} | {ceil_print(REP0d3, 3)} '
              f'| {ceil_print(REP0d4, 3)} | {May_REP0:5.3f} '
              f'| {ceil_print(REP1d2, 3)} | {ceil_print(REP1d3, 3)} '
              f'| {ceil_print(REP1d4, 3)} | {May_REP1:5.3f} |')
    print()


def plot_algorithms(max_depth=3):
    omegas = list(np.arange(0.001, 0.1, 0.001)) + list(np.arange(0.1, 1, 0.01))
    depths = list(range(2, max_depth + 1))

    labels = [
        'Odlyzko', 'Howgrave-Graham',
        *[f'REP-0 (d={d}),' for d in depths],
        *[f'REP-1 (d={d}),' for d in depths],
    ]
    data = []

    with open(f'may_asymptotics_depth{max_depth}.csv', 'w') as f:
        print('omega, lg |S|, Odlyzko, Howgrave-Graham,',
              *[f'REP-0 (d={d}),' for d in depths],
              *[f'REP-1 (d={d}),' for d in depths], file=f)
        for omega in omegas:
            ss = lgS(omega)
            odly = odlyzko(omega)
            hw = howgrave_graham(omega)

            REP0 = [meet_rep0(d, omega) for d in depths]
            REP1 = [meet_rep1(d, omega) for d in depths]

            data.append([
                odly / ss, hw / ss,
                *[t / ss for t in REP0],
                *[t / ss for t in REP1],
            ])

            print(f'{omega:5.3f}, {lgS(omega):6.4f}, {odlyzko(omega):7.4f}, '
                  f'{howgrave_graham(omega):15.4f},',
                  *[f'{t:11.4f},' for t in REP0],
                  *[f'{t:11.4f},' for t in REP1],
                  file=f)

    data = list(zip(*data))
    for label, ys in zip(labels, data):
        plt.plot(omegas, ys, label=label)
    plt.xlabel('omega ~ w/n')
    plt.ylabel('log(time) / log(search-space)')
    plt.gca().set_ylim([0.2, 0.8])
    plt.legend()

    # Generate 1920x1080 image:
    plt.gcf().set_size_inches(16, 9)
    plt.savefig(f'may_asymptotics_depth{max_depth}.png', dpi=120)

    plt.show()


def __main__():
    max_depth = 3 if len(sys.argv) == 1 else int(sys.argv[1])
    # verify_table_5()
    plot_algorithms(max_depth)


if __name__ == "__main__":
    __main__()
