**Title**: Add a Full-Form IPv6 Regex Type

Define and export `ipv6` from `outlines.types` as a `Regex` DSL term. This type
covers the full eight-group textual subset: exactly eight colon-separated groups,
each containing one to four ASCII hexadecimal digits. Hex letters may be upper-
or lowercase. Empty groups, `::` compression, zone identifiers, IPv4 tails, and
groups longer than four digits are invalid. No claim of complete IPv6 notation
support is intended.
