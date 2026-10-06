**Title**: Add an `IsIP` validator

Add a public `IsIP` validator for IP addresses and networks. It must be
available from the top-level `dirty_equals` package and follow the equality and
assertion behavior of the library's existing validators.

**Public contract**:

- The constructor is `IsIP(version=None, *, netmask=None, within=None)`; `version` may be
  supplied positionally and is `None`, `4`, or `6`.
- Accept valid IPv4 and IPv6 address or network values represented as strings,
  integers, packed bytes, or existing `ipaddress` address/network objects.
- When `version` is given, reject values of the other version.
- Integer candidates in `0 <= value < 2**32` represent IPv4 addresses; those
  in `2**32 <= value < 2**128` represent IPv6 addresses. `version` only filters
  this independently determined address family; it does not reinterpret the integer.
- When `netmask` is given, accept only networks with that mask. Supplying a
  netmask without a version raises `TypeError` with the message
  `To check the netmask you must specify the IP version`.
- `netmask` accepts an IPv4 dotted-decimal mask, an IPv6 mask string, or the
  corresponding integer mask value. An integer denotes the complete mask
  value, not a CIDR prefix length.
- Invalid values compare unequal rather than leaking parsing exceptions.
- `repr(IsIP())` is `IsIP()`.

`within` restricts matching to a strict CIDR network string, an `ipaddress`
network object, or a nonempty list/tuple of these. An address must belong to
at least one configured network; an input network must be wholly contained
in one configured network (overlap alone is insufficient). Versions must
match for containment. This restriction combines with version and netmask,
not instead of them. None disables the restriction. Invalid or empty range
configuration raises ValueError at construction, including CIDRs with host
bits set, bare address strings without a slash, and non-string/non-network
range elements. Candidate network inputs retain the existing host-bit normalization.
Successful matching retains the original candidate as the validator's value.
