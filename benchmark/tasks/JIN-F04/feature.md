**Title**: Limit the number of groups returned by groupby

Add max_groups=None to groupby. None returns every group; a nonnegative integer
returns the first N groups in the usual ascending key order, including every
member of each retained group. Zero returns no groups. This is not a ranking by
group size or a limit on input items. Negative, noninteger and boolean values
raise FilterArgumentError. Preserve existing attribute/default behavior and
synchronous/asynchronous execution, including one-shot and async inputs.
