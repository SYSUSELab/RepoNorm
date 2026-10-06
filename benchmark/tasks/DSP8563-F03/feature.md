# Minimal scalar and unit coercion for Tool arguments

Before validating arguments, let Tool accept common scalar strings according to
the declared JSON schema. Apply this through both ordinary Tool calls and acall;
preserve existing schema constraints and Pydantic argument parsing afterward.

For boolean fields, accept case-insensitive "true" and "false". For integer and
number fields, accept finite decimal numeric strings, with optional sign, decimal
point and exponent. Preserve exact integers, including those beyond binary-float
precision; nonintegral values must still fail an integer schema.

Numeric fields also accept nonnegative decimal durations with suffix ms, s, m or h
and sizes with suffix KB, MB or GB. Units are case-sensitive; allow whitespace
between magnitude and suffix. Convert durations to seconds and sizes to integer
bytes using decimal factors (KB=1000), truncating fractional bytes toward zero.
Unit magnitudes do not use exponent notation. Recognized duration/bytes formats
may enable unit parsing when type is unspecified, but string fields stay strings.

Trim surrounding whitespace for conversion. Only convert supplied strings; leave
string-typed fields, unknown **kwargs and non-string values unchanged. Unknown
units, NaN/Inf, expressions and failed conversions retain the original value for
the existing schema validation error path. Do not execute input text or mutate
caller-owned input mappings. Keep defaults and existing function signatures.
