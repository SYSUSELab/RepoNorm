**Title**: Hierarchical groupby results

Add levels=None to groupby. With an explicit positive integer levels, accept an
attribute list or a comma-separated string (surrounding whitespace ignored), whose
length must equal levels. Each attribute defines one level, in order. Each result
group retains the grouper/list interface; intermediate lists contain child groups
and the final lists contain the original items. Each input item must occur in
exactly one leaf; grouping and member ordering apply consistently at every level.

For example groupby(['department', 'role'], levels=2) returns departments whose
lists contain role groups. Support numeric and dotted attribute paths and the
existing default at every level without mutating the input or attribute list.
Synchronous/asynchronous execution and one-shot/async iterators remain supported.

Omitting levels keeps the ordinary single-attribute behavior; a one-element
attribute list is also accepted. Multiple attributes without levels, mismatched
attribute counts, and nonpositive/noninteger/boolean levels raise FilterArgumentError.
