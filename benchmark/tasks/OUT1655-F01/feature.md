**Title**: Add Hexadecimal, UUID4, and IPv4 Regex Types

Define and export `hex_str`, `uuid4`, and `ipv4` from `outlines.types`.
Each value must be an `outlines.types.Regex` term so it works with the existing
regex DSL, Pydantic schemas, and term composition.

The complete matched value follows these rules:

- `hex_str` contains one or more ASCII hexadecimal digits (`0-9`, `a-f`, or
  `A-F`), optionally preceded by the exact lowercase prefix `0x`. A bare prefix,
  uppercase `0X`, whitespace, and non-hexadecimal characters are invalid.
- `uuid4` uses the canonical hyphenated 8-4-4-4-12 UUID layout and ASCII
  hexadecimal digits. The version nibble is `4`; the variant nibble is one of
  `8`, `9`, `a`, `A`, `b`, or `B`. Braces, missing hyphens, and surrounding
  whitespace are invalid.
- `ipv4` has exactly four dot-separated octets. Each octet has one to three ASCII
  decimal digits, may contain leading zeroes, and has a numeric value from 0
  through 255 inclusive. Therefore `00.01.10.99` and `000.001.025.255` are valid,
  while four-digit octets and numeric values above 255 are invalid.

Any equivalent regular-expression construction is acceptable. Existing predefined
types and DSL behavior must continue to work.
