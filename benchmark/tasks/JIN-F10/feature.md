**Title**: Filter complete groups with synchronous or asynchronous predicates

Add filter_groups=None to groupby. A predicate receives the complete member list
of each group once; retain that group only when its result is truthy. None retains
all groups. Preserve group order, member order, attribute/default resolution and
the group-result interface. Empty input does not invoke the predicate.

Ordinary callbacks and @pass_environment callbacks are supported in both modes;
the latter receive the active Environment before the member list. In async
environments, await predicate results when necessary while still accepting
synchronous return values and asynchronous input iterators. Synchronous mode need
not execute async predicates. Exceptions propagate unchanged. No concurrent
callback execution is required; awaitable false must not be treated as truthy.
