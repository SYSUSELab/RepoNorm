**Title**: Exclude selected token IDs from list and NumPy encoding

Add filter_tokens=None to encode and encode_to_numpy. A list, tuple or set of token
IDs removes all occurrences of those IDs after encoding; duplicates have no extra
effect. Preserve remaining token order. None or an empty collection performs no
filtering. NumPy results remain one-dimensional uint32 arrays, including when all
tokens are removed. Caller collections must not be mutated.

For valid Unicode input, the NumPy values must match the corresponding list call
under the same filtering and special-token policy, including disallowed_special=()
treating special literals as ordinary text. A prohibited special literal still
raises before filtering, even if its token ID would be removed. Do not require a
particular private buffer or zero-copy strategy. The list API's existing surrogate
fallback remains supported; adding such a fallback to NumPy is outside scope.
