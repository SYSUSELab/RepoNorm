**Title**: Configurable attribute-path separator in groupby and map

Support separator='.' in groupby and map(attribute=...). A nonempty custom
separator, including multiple characters, replaces the dot delimiter for that
call; ordinary dots then become literal field-name characters. Preserve existing
attribute traversal through objects, mappings and numeric indices, default
substitution, synchronous/asynchronous execution and one-shot/async inputs.
For example p__0__literal.key with separator='__' accesses the literal.key field
of the first item under p. Both filters must agree on path interpretation.

An empty separator raises FilterArgumentError when the attribute path is evaluated
(map remains lazy for empty inputs). The map-by-filter-name mode must
continue forwarding all arguments, including separator, to the named filter.
Other filters and calls omitting separator retain their existing behavior.
