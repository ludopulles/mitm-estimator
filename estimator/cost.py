# -*- coding: utf-8 -*-
from collections import UserDict

from sage.all import log, oo, round


# UserDict inherits from typing.MutableMapping
class Cost(UserDict):
    """
    Algorithms costs.
    """

    # An entry is "impermanent" if it grows when we run the algorithm again. For example, `δ`
    # would not scale with the number of operations but `rop` would. This check is strict such that
    # unknown entries raise an error. This is to enforce a decision on whether an entry should be
    # scaled.

    impermanents = {
        # Timing related:
        "rop": True,  # number of ring operations
        "red": True,  # rop's spent on lattice reduction
        "svp": True,  # rop's spent to solve SVP

        # General properties:
        "prob": False,  # success probability
        "repetitions": False,  # number of repetitions required to boost success probability
        "tag": False,  # name of the attack
        "problem": False,  # targeted LWE parameters
        "mem": False,  # memory usage

        # Specific attack parameters:
        "zeta": False,  # guessing dimension
        "d": False,  # projected dimension, after ignoring initial part of basis
        "beta": False,  # Block size for BKZ
        "delta": False,  # n-th root Hermite factor, a lattice reduction quality measure
        "|S|": False,  # Search space (size)
        "eta": False,  # dimension of Babai nearest plane (see lwe_primal.py)
        "h_": False,  # guessed hamming weight of guessing part of a sparse secret
    }

    @staticmethod
    def _update_without_overwrite(dst, src):
        keys_intersect = set(dst.keys()) & set(src.keys())
        attempts = [
          f"{k}: {dst[k]} with {src[k]}" for k in keys_intersect if dst[k] != src[k]
        ]
        if len(attempts) > 0:
            s = ", ".join(attempts)
            raise ValueError(f"Attempting to overwrite {s}")
        dst.update(src)

    @classmethod
    def register_impermanent(cls, data=None, **kwds):
        if data is not None:
            cls._update_without_overwrite(cls.impermanents, data)
        cls._update_without_overwrite(cls.impermanents, kwds)

    key_map = {
        "delta": "δ",
        "beta": "β",
        "eta": "η",
        "epsilon": "ε",
        "zeta": "ζ",
        "ell": "ℓ",
        "repetitions": "↻",
    }

    # Note: these are in the range of U+2080 - U+208E.
    # Superscripts are found in U+00B2, U+00B3, U+00B9, and U+2070 - U+207F.
    key_sub_map = {
        "": "'",
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    }

    val_map = {"beta": "%8d", "beta_": "%8d", "d": "%8d", "delta": "%8.6f"}

    def str(self, keyword_width=0, newline=False, round_bound=2048, compact=False):
        """

        :param keyword_width:  keys are printed with this width
        :param newline:        insert a newline
        :param round_bound:    values beyond this bound are represented as powers of two
        :param compact:        do not add extra whitespace to align entries

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> s = Cost(delta=5, bar=2)
            >>> s
            δ: 5.000000, bar: 2

        """
        def format_key(key):
            if key.find("_") >= 0:
                key, sub = key.split("_")
                return self.key_map.get(key, key) + self.key_sub_map.get(sub, "_" + sub)
            return self.key_map.get(key, key)

        def value_str(k, v):
            kstr = format_key(k)
            kk = f"{kstr:>{keyword_width}}"
            try:
                if (1 / round_bound < abs(v) < round_bound) or (not v) or (k in self.val_map):
                    if abs(v % 1) < 1e-7:
                        vv = self.val_map.get(k, "%8d") % round(v)
                    else:
                        vv = self.val_map.get(k, "%8.3f") % v
                else:
                    vv = "%7s" % ("≈2^%.1f" % log(v, 2))
            except TypeError:  # strings and such
                vv = "%8s" % v
            if compact is True:
                kk = kk.strip()
                vv = vv.strip()
            return f"{kk}: {vv}"

        # we store the problem instance in a cost object for reference
        s = [value_str(k, v) for k, v in self.items() if k != "problem"]
        delimiter = "\n" if newline is True else ", "
        return delimiter.join(s)

    def reorder(self, *args):
        """
        Return a new ordered dict from the key:value pairs in dictionary but reordered such that the
        keys given to this function come first.

        :param args: keys which should come first (in order)

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> d = Cost(a=1,b=2,c=3); d
            a: 1, b: 2, c: 3

            >>> d.reorder("b","c","a")
            b: 2, c: 3, a: 1

        """
        reord = {k: self[k] for k in args if k in self.keys()}
        reord.update(self)
        return Cost(**reord)

    def filter(self, **keys):
        """
        Return new ordered dictionary from dictionary restricted to the keys.

        :param dictionary: input dictionary
        :param keys: keys which should be copied (ordered)
        """
        r = {k: self[k] for k in keys if k in self.keys()}
        return Cost(**r)

    def repeat(self, times, select=None):
        """
        Return a report with all costs multiplied by ``times``.

        :param times:  the number of times it should be run
        :param select: toggle which fields ought to be repeated and which should not
        :returns:      a new cost estimate

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> c0 = Cost(a=1, b=2)
            >>> c0.register_impermanent(a=True, b=False)
            >>> c0.repeat(1000)
            a: 1000, b: 2, ↻: 1000

        TESTS::

            >>> from estimator.cost import Cost
            >>> Cost(rop=1).repeat(1000).repeat(1000)
            rop: ≈2^19.9, ↻: ≈2^19.9

        """
        impermanents = dict(self.impermanents)

        if select is not None:
            impermanents.update(select)

        try:
            new_cost = {k: times * v if impermanents[k] else v for k, v in self.items()}
            new_cost["repetitions"] = times * new_cost.get("repetitions", 1)
            return Cost(**new_cost)
        except KeyError as error:
            raise NotImplementedError(
                f"You found a bug, this function does not know about about a key but should: {error}"
            )

    def __rmul__(self, times):
        return self.repeat(times)

    def combine(self, right):
        """Combine ``self`` and ``right`` into one cost.

        :param self: this cost dictionary
        :param right: cost dictionary

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> c0, c1, c2 = Cost(a=1), Cost(b=2), Cost(c=3)
            >>> c0.combine(c1)
            a: 1, b: 2
            >>> c2.combine(c0).combine(c1)
            c: 3, a: 1, b: 2

        """
        return Cost(**self, **right)

    def __bool__(self):
        return self.get("rop", oo) < oo

    def __add__(self, other):
        """
        Return the combination of this cost and cost of ``other``.

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> c0, c1, c2 = Cost(a=1), Cost(b=2), Cost(c=3)
            >>> c1 += c0
            >>> c2 + c1
            c: 3, b: 2, a: 1

        """
        return self.combine(other)

    def __repr__(self):
        return self.str(compact=True)

    def __str__(self):
        return self.str(newline=True, keyword_width=12)

    def __lt__(self, other):
        try:
            return self["rop"] < other["rop"]
        except AttributeError:
            return self["rop"] < other

    def __le__(self, other):
        try:
            return self["rop"] <= other["rop"]
        except AttributeError:
            return self["rop"] <= other

    def sanity_check(self):
        """
        Perform basic checks.
        """
        if self.get("rop", 0) > 2**10000:
            self["rop"] = oo
        if self.get("beta", 0) > self.get("d", 0):
            raise RuntimeError(f"β = {self['beta']} > d = {self['d']}")
        if self.get("eta", 0) > self.get("d", 0):
            raise RuntimeError(f"η = {self['eta']} > d = {self['d']}")
        return self
