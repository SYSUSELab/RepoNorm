# Contributor rules

Public API boundary functions must translate `InternalStorageError` into
`AuroraPublicError`. Internal helpers may propagate `InternalStorageError`.
