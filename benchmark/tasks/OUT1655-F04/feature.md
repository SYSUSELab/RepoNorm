**Title**: Add a MAC Address Regex Type

Define and export `mac_address` from `outlines.types` as a `Regex` DSL term.
It matches exactly six pairs of ASCII hexadecimal digits. The five separators
must all be colons or must all be hyphens; mixed separators are invalid. Hex
letters may be upper- or lowercase. Dotted, unseparated, shortened, EUI-64, and
surrounded-by-whitespace forms are outside this format.

```python
from outlines import types

MacModel = types.mac_address
```
