**Title**: Character chunks with configurable overlap

Add chunk_size=None and chunk_overlap=0 to encode. With a positive integer
chunk_size, return independently encoded character windows. Adjacent windows
overlap by chunk_overlap characters, which must be an integer satisfying
0 <= chunk_overlap < chunk_size. Stop at the first window covering the end; do
not emit redundant trailing windows. For example 'abcde', size 3, overlap 1 gives
['abc', 'cde']. Empty text yields [[]]. These are Python character windows, not
byte or token windows; concatenating token windows need not equal whole-text BPE.

Without chunk_size, return ordinary encoding and require overlap zero. Invalid
sizes, overlaps and combinations raise ValueError; bool is not an integer option.
Apply the existing disallowed-special policy to the complete input before splitting.
Allowed special literals wholly within a window encode normally; if cut by a
boundary, each window's text is encoded independently. Preserve ordinary Unicode
fallback on the nonchunking path.
