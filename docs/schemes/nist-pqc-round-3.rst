NIST PQC Round 3 Finalists
==========================

`Kyber <https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf>`__

::

    >>> from estimator import *
    >>> schemes.Kyber512
    ModuleLWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=1.22), m=512, tag='Kyber 512', ringdeg=256, rank=2)
    >>> LWE.primal_bdd(schemes.Kyber512)
    rop: ≈2^144.2, red: ≈2^143.2, svp: ≈2^143.1, β: 404, η: 407, d: 996, tag: bdd

::

    >>> from estimator import *
    >>> schemes.Kyber768
    ModuleLWEParameters(n=768, q=3329, Xs=D(σ=1.00), Xe=D(σ=1.00), m=768, tag='Kyber 768', ringdeg=256, rank=3)
    >>> LWE.primal_bdd(schemes.Kyber768)
    rop: ≈2^205.0, red: ≈2^203.8, svp: ≈2^204.1, β: 620, η: 626, d: 1440, tag: bdd

::

    >>> from estimator import *
    >>> schemes.Kyber1024
    ModuleLWEParameters(n=1024, q=3329, Xs=D(σ=1.00), Xe=D(σ=1.00), m=1024, tag='Kyber 1024', ringdeg=256, rank=4)
    >>> LWE.primal_bdd(schemes.Kyber1024)
    rop: ≈2^274.9, red: ≈2^273.8, svp: ≈2^274.0, β: 869, η: 876, d: 1898, tag: bdd

`Saber <https://www.esat.kuleuven.be/cosic/pqcrypto/saber/files/saberspecround3.pdf>`__

::

    >>> from estimator import *
    >>> schemes.LightSaber
    ModuleLWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=2.29, μ=-0.50), m=512, tag='LightSaber', ringdeg=256, rank=2)
    >>> LWE.primal_bdd(schemes.LightSaber)
    rop: ≈2^143.8, red: ≈2^142.7, svp: ≈2^142.8, β: 402, η: 406, d: 1022, tag: bdd

::

    >>> from estimator import *
    >>> schemes.Saber
    ModuleLWEParameters(n=768, q=8192, Xs=D(σ=1.41), Xe=D(σ=2.29, μ=-0.50), m=768, tag='Saber', ringdeg=256, rank=3)
    >>> LWE.primal_bdd(schemes.Saber)
    rop: ≈2^212.1, red: ≈2^211.1, svp: ≈2^211.1, β: 646, η: 651, d: 1489, tag: bdd

::

    >>> from estimator import *
    >>> schemes.FireSaber
    ModuleLWEParameters(n=1024, q=8192, Xs=D(σ=1.22), Xe=D(σ=2.29, μ=-0.50), m=1024, tag='FireSaber', ringdeg=256, rank=4)
    >>> LWE.primal_bdd(schemes.FireSaber)
    rop: ≈2^279.8, red: ≈2^278.8, svp: ≈2^278.7, β: 887, η: 893, d: 1926, tag: bdd


`NTRU <https://ntru.org/f/ntru-20190330.pdf>`__

::

    >>> from estimator import *
    >>> schemes.NTRUHPS2048509Enc
    NTRUParameters(n=508, q=2048, Xs=D(σ=0.82), Xe=T(hw=254, ones=127, n=508), m=508, tag='NTRUHPS2048509Enc', ntru_type='matrix')
    >>> NTRU.primal_bdd(schemes.NTRUHPS2048509Enc)
    rop: ≈2^131.1, red: ≈2^130.1, svp: ≈2^130.1, β: 357, η: 389, d: 914, tag: bdd

::

    >>> from estimator import *
    >>> schemes.NTRUHPS2048677Enc
    NTRUParameters(n=676, q=2048, Xs=D(σ=0.82), Xe=T(hw=254, ones=127, n=676), m=676, tag='NTRUHPS2048677Enc', ntru_type='matrix')
    >>> NTRU.primal_bdd(schemes.NTRUHPS2048677Enc)
    rop: ≈2^170.7, red: ≈2^169.6, svp: ≈2^169.9, β: 498, η: 532, d: 1177, tag: bdd

::

    >>> from estimator import *
    >>> schemes.NTRUHPS4096821Enc
    NTRUParameters(n=820, q=4096, Xs=D(σ=0.82), Xe=T(hw=510, ones=255, n=820), m=820, tag='NTRUHPS4096821Enc', ntru_type='matrix')
    >>> NTRU.primal_bdd(schemes.NTRUHPS4096821Enc)
    rop: ≈2^199.6, red: ≈2^198.6, svp: ≈2^198.6, β: 601, η: 635, d: 1482, tag: bdd

::

    >>> from estimator import *
    >>> schemes.NTRUHRSS701Enc
    NTRUParameters(n=700, q=8192, Xs=D(σ=0.82), Xe=D(σ=0.82), m=700, tag='NTRUHRSS701', ntru_type='matrix')
    >>> NTRU.primal_bdd(schemes.NTRUHRSS701Enc)
    rop: ≈2^158.6, red: ≈2^157.6, svp: ≈2^157.6, β: 454, η: 489, d: 1306, tag: bdd
