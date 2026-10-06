**Title**: Descending group order for groupby

Add reverse=False to groupby. When true, groups are ordered by their keys
descending instead of ascending; members within each group retain input order.
Support synchronous and asynchronous execution, one-shot and async inputs, empty
inputs, existing attribute resolution (including nested numeric paths), default,
and the existing group-result interface. Keys must remain mutually comparable;
no new ordering between unrelated types is required.
