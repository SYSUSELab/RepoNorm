**Title**: Add a SHA-256 Hash Format Type

Define and export `hash_sha256` from `outlines.types` as a `Regex` DSL term. It
matches exactly 64 ASCII hexadecimal characters; letters may be upper- or
lowercase. Whitespace, prefixes, separators, non-hexadecimal characters, and
other lengths are invalid. This type validates textual shape only and does not
verify that the value is the digest of any claimed payload. It must retain normal
Pydantic schema and regex DSL composition behavior.
