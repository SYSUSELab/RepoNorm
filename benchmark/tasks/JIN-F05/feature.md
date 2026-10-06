**Title**: Control empty grouping keys

Add include_empty=True to groupby in synchronous and asynchronous environments.
False removes items whose resolved grouping key is falsy, after existing default
substitution. True retains them. Define the retained-key order as normal comparable
keys first (including the empty string in string order), then None, then a separate
missing/Undefined group. For example keys [missing, None, '', 'z', missing, None]
produce groups ['', 'z', None, missing] of sizes [1, 1, 2, 2]. Missing keys must
not split into multiple groups when interleaved with None. StrictUndefined retains
its error behavior rather than silently becoming an ordinary missing key.

Apart from the explicit None/Undefined rules, retained keys must be mutually
orderable, even when falsy (for example mixing '' and 0 is outside scope).
No new cross-type ordering is required for other unrelated types. Preserve
attribute/default resolution, result-group interfaces, member order, and one-shot
or asynchronous inputs. This explicitly extends empty-key ordering; it does not
claim the old implementation already sorted None alongside strings.
