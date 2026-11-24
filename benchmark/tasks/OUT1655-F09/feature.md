**Title**: Add a Braced GUID Regex Type

Define and export `guid` from `outlines.types` as a `Regex` DSL term. It matches
an opening brace, five hyphen-separated ASCII hexadecimal segments of lengths
8-4-4-4-12, and a closing brace. Hex letters may be upper- or lowercase. Both
braces are required. This format imposes no UUID version or variant restriction;
unbraced, parenthesized, malformed-length, and whitespace-surrounded forms are
invalid.
