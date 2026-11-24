**Title**: Add IsCreditCard validator with Luhn algorithm and issuer detection

Add a public `IsCreditCard` validator that follows dirty-equals' existing
equality, assertion, and top-level import behavior. It validates card numbers
with the Luhn algorithm and optionally restricts them to a named issuer.

**Public contract**:

- Accept card numbers as strings or integers. Strings may contain spaces,
  hyphens, and surrounding whitespace; other characters and input types compare
  unequal.
- Without an issuer, any 12--19 digit number that passes the Luhn check is
  accepted.
- Digits are ASCII `0` through `9`, not Unicode numeric characters. No
  real-world card allocation, BIN database or payment capability is checked.
- The optional issuer is exactly one of `Visa`, `Mastercard`, or
  `AmericanExpress`, matched case-sensitively.
- Visa numbers start with 4 and contain 13, 16, or 19 digits.
- Mastercard numbers contain 16 digits and start in the 51--55 or 2221--2720
  ranges.
- American Express numbers contain 15 digits and start with 34 or 37.
- A number with an issuer mismatch, invalid prefix or length, or invalid Luhn
  checksum compares unequal.
