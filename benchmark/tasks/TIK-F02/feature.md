**Title**: Unicode token positions and batch alignment

Add return_positions=False to encode and encode_batch. True returns (tokens,
positions) for each encoded document; a batch returns a list of those pairs in
input order. Token IDs must exactly match ordinary encoding under the same special
token policy. positions are zero-based Python character indices in each original
document, not byte indices. If a token starts inside a multibyte character, use the
index of that containing character; repeated positions are valid. Special tokens
start at their literal text's starting character. Empty input yields ([], []).
Support valid Unicode, including emoji and combining characters, and ordinary
allowed/disallowed special-token behavior. Position mapping of surrogate-containing
strings is outside scope; ordinary non-position encoding must retain its existing
surrogate fallback. Default return types remain unchanged.
