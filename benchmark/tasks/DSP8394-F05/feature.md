# Optional disk cache compression

Add `compress=None`, `cull_limit=None` and `eviction_policy=None` to Cache and
`dspy.clients.configure_cache`, keeping existing defaults. Compression accepts
`"gzip"`, `"zlib"` or None; reject other codecs with ValueError. Expose
`cache.set_compression(codec)` to change the codec for subsequent disk writes.
An invalid update leaves the previous configuration usable.

Compression applies only to the disk layer. Both supported codecs must preserve
original value types and nested data on round trip, and measurably reduce storage
for large compressible responses. Keep ordinary memory-cache behavior, copy-on-read
and usage clearing. Memory-only and disabled-cache configurations remain usable.

The persisted compressed representation is bytes beginning with `b"DSPC"`, followed
by one codec byte (0x00: uncompressed pickle; 0x01: gzip; 0x02: zlib) and the encoded
pickle payload. Reading uses the stored header, independently of the currently
selected write codec. Existing encoded entries must remain readable after switching
to a different codec or None, including from another Cache instance on the same store.
The header is a reserved format marker; unknown codec selectors raise ValueError.

Retain compatibility with actual pre-feature DiskCache entries: header-less values
have already been deserialized by the backend and must keep their Python value,
including ordinary bytes. Do not unpickle arbitrary header-less byte values.
Uncompressed writes may keep the existing backend representation.

Apply cull_limit and eviction_policy to the disk backend, preserving its normal
semantics and defaults when omitted. No change to eviction behavior is required
for the in-memory layer.
