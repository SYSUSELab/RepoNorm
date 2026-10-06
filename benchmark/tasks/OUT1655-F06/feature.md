**Title**: Add a Restricted HTTP URL Regex Type

Define and export `url` from `outlines.types` as a `Regex` DSL term for this
explicit URL subset:

- the scheme is exactly lowercase `http` or `https`;
- the host is `localhost`, a DNS name with at least two labels, or a four-octet
  IPv4 address;
- DNS labels contain ASCII letters, digits, or internal hyphens, begin and end
  with an alphanumeric character, and contain at most 63 characters;
- IPv4 octets use canonical decimal spelling without leading zeroes and have a
  numeric value from 0 through 255;
- an optional port contains one to five ASCII decimal digits;
- an optional slash-led path may contain ASCII URL punctuation but no whitespace;
- optional query and fragment components use `?` and `#` respectively and must
  contain at least one permitted non-whitespace character when present.

User information, uppercase schemes, empty host labels, bracketed IPv6 hosts,
and whitespace are invalid. This is a practical subset and is not RFC-complete.
