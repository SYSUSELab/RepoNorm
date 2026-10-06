**Title**: Add case-sensitive control to groupby

Extend groupby with case_sensitive=False in synchronous and asynchronous environments.
String keys differing only by lowercasing belong to one group by default, retaining
the spelling of the first encountered key and the input order of its members.
case_sensitive=True keeps differently cased keys separate. Non-string comparable
keys retain normal grouping behavior. Unicode handling follows lowercasing, not
casefolding. Existing attribute paths (including numeric parts), default values,
one-shot input iterators, and the existing group-result interface remain supported.
Async environments also accept asynchronous input iterators.
