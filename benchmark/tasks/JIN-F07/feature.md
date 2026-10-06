**Title**: First-seen group order, including unhashable keys

Add stable=False to groupby. True orders groups by the first occurrence of each
key instead of sorting keys. Members retain input order. Equal keys merge even
when not adjacent or not the same object; retain the first original key object
as grouper. Support equality-comparable keys including lists and dictionaries,
and mixtures with hashable keys, without requiring ordering or hashing or changing
the keys. String representations must not define key equality.

False retains existing sorted grouping. Support existing attribute/default
resolution, one-shot iterators and synchronous/asynchronous environments including
async inputs. Exceptional or nonreflexive equality and a universal linear-time
bound are outside scope. The old sorted grouping already preserves member order;
the new behavior concerns group order and nonsortable equality-comparable keys.
