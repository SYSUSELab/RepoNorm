**Title**: Token frequency summaries for text and corpora

Add analyze_frequency=False to encode and encode_batch. True on encode returns
(tokens, frequency), where frequency maps token IDs to occurrence counts. True on
encode_batch returns (document_token_lists, corpus_frequency), with counts summed
across every document, including repeated documents. Empty batch returns ([], {}).
Keep per-document order, vocabulary isolation, Unicode and special-token policies.
Ordinary calls retain their existing return types and encoding contents. Batch
thread count must not change the summary. This is a functional summary feature;
no specific internal counting technique or performance improvement is required.
