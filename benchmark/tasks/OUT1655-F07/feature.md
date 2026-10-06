**Title**: Add a Credit Card Number Format Type

Define and export `credit_card` from `outlines.types` as a `Regex` DSL term.
It accepts exactly 16 ASCII digits in one of three shapes: no separators, four
groups of four separated by single hyphens, or four groups of four separated by
single spaces. Separators must be consistent. This type checks textual shape only;
it does not validate Luhn checksums, issuers, BIN ranges, or whether a number was
issued. Shape-valid values remain valid regardless of checksum.
