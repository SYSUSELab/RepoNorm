**Title**: Token budgets for individual and batch encoding

Add max_tokens=None to encode and encode_batch, and max_total_tokens=None to
encode_batch. Every explicit limit must be a positive integer (not bool), otherwise
raise ValueError. Validate these limits even for empty input; empty input with
valid limits remains allowed. max_tokens limits each document;
max_total_tokens limits the sum across the batch. The limits apply independently
and together; exact equality is accepted. An over-budget result raises ValueError
rather than returning a partial batch. Per-document budget errors identify the
zero-based index of the first offending document in input order; total-budget
errors identify max_total_tokens. When both are exceeded, report the document
violation first. Ordinary encoding content, order, Unicode handling and special-token
policy remain unchanged. These are output token budgets, not an early BPE stopping
or peak-memory guarantee.
