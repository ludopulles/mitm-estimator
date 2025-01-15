# -*- coding: utf-8 -*-
"""
Default values.
"""

from .simulator import GSA
from .reduction import RC
from sage.all import exp

"""
Default models used to evaluate the cost and shape of lattice reduction.
This influences the concrete estimated cost of attacks.
"""
red_cost_model = RC.MATZOV
red_cost_model_classical_poly_space = RC.ABLR21
red_shape_model = "gsa"
red_simulator = GSA

mitm_opt = "analytical"
max_n_cache = 10000

# Best time to multiply two nxn matrices is O(n^{exponent_matmul}).
# Naive matrix multiplication is O(n^3), but Strassen proved exponent_matmul < 2.81 in 1969 [1].
# The constant 2.371552 was best known at the time of January 2024 [2].
# [1] Volker Strassen.
#     Gaussian elimination is not optimal.
#     Numerische Mathematik 13, 354–356 (1969).
#     https://doi.org/10.1007/BF02165411
# [2] Vassilevska Williams, Virginia; Xu, Yinzhan; Xu, Zixuan; Zhou, Renfei.
#     New Bounds for Matrix Multiplication: from Alpha to Omega.
#     Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). pp. 3792–3835.
#     arXiv:2307.07970. https://doi.org/10.1137/1.9781611977912.134
exponent_matmul = 2.371552


def ntru_fatigue_lb(n):
    return int((n**2.484)/exp(6))
