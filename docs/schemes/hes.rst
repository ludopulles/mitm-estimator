Homomorphic Encryption Parameters
=================================

::

    >>> from estimator import *
    >>> from estimator.schemes import HESv111024128error
    >>> HESv111024128error
    ModuleLWEParameters(n=1024, q=134217728, Xs=D(σ=3.00), Xe=D(σ=3.00), m=1024, tag='HESv11error', ringdeg=1024, rank=1)
    >>> LWE.primal_bdd(HESv111024128error)
    rop: ≈2^139.2, red: ≈2^138.9, svp: ≈2^136.8, β: 383, η: 377, d: 2043, tag: bdd

::

    >>> from estimator import *
    >>> from estimator.schemes import HESv111024128ternary
    >>> HESv111024128ternary
    ModuleLWEParameters(n=1024, q=134217728, Xs=D(σ=0.82), Xe=D(σ=3.00), m=1024, tag='HESv11ternary', ringdeg=1024, rank=1)
    >>> LWE.primal_hybrid(HESv111024128ternary)
    rop: ≈2^188.1, red: ≈2^184.7, svp: ≈2^188.0, β: 345, η: 2, ζ: 131, |S|: ≈2^199.5, d: 1881, prob: ≈2^-54.3, ↻: ≈2^56.5, h': 72, tag: hybrid
   
::

    >>> from estimator import *
    >>> from estimator.schemes import SEAL22_8192
    >>> SEAL22_8192
    ModuleLWEParameters(n=8192, q=107839786668602559178668060348078522694548577690162289924414373888001, Xs=D(σ=0.82), Xe=D(σ=3.19), m=+Infinity, tag='SEAL22_8192', ringdeg=8192, rank=1)
    >>> LWE.dual_hybrid(SEAL22_8192)
    rop: ≈2^121.8, red: ≈2^121.8, guess: ≈2^101.7, β: 306, p: 3, ζ: 10, t: 40, β': 331, N: ≈2^68.1, m: ≈2^13.0
