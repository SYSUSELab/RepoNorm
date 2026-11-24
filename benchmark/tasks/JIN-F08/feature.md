**Title**: Count metadata on groupby results

Add with_counts=False to groupby. True returns each group as a three-field result
supporting (grouper, items, count) unpacking/indexing and .grouper, .list, .count
data attributes in Python and templates. count records the number of members when
the result is created; it is not a tuple count(value) method on this new result.
False retains the original two-field group result and its behavior unchanged.

Preserve grouping, ordering, attribute/default paths, empty input, one-shot
iterators and synchronous/asynchronous execution including asynchronous inputs.
Counts must describe the complete returned member list, not a consumed iterator.
