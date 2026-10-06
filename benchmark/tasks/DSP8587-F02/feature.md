**Title**: Configure the Streaming End Buffer and Report Its Watermark

Add `end_buffer_size: int = 10` to `StreamListener`. It controls the number of
recent model chunks retained for end-identifier detection. The value must be a
plain integer from 3 through 64 inclusive; booleans are not integers for this
API. Invalid values raise `ValueError` with these exact messages:

- non-integer: `end_buffer_size must be an integer`
- below 3: `end_buffer_size must be at least 3`
- above 64: `end_buffer_size must be at most 64`

Replace the hard-coded buffer size with this setting without changing adapter
boundaries or emitted content.

Add `buffer_watermark: int` to `StreamResponse`. Each emitted response reports
`min(current_queue_size, end_buffer_size)`, where `current_queue_size` is
captured after the received model chunk is enqueued and before that emission
dequeues or flushes buffered chunks. This rule applies to ordinary and final
emissions. The default buffer size remains 10.

