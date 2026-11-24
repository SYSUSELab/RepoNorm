**Title:** Add stateless translation extraction markers

The `TranslationExtractorExtension` provides `{% extract key=constant, ... %}...{% endextract %}`. Metadata values must be compile-time constants; dynamic expressions and duplicate keys raise `TemplateSyntaxError` naming the key. Marker bodies are ordinary Jinja bodies and render unchanged. They may contain control structures and any gettext calls supported by normal extraction.

`Environment.extract_marked_translations(source, source_name=None)` parses a string or accepts a parsed template and yields independent record mappings. Each record has `source`, marker `lineno`, a copied `metadata` mapping, and `messages`. Each message contains its `lineno`, gettext `function`, and extracted `message`. Nested gettext calls in the marker body are associated with that record.

Extraction retains no records between calls or template compilations, so recompiling a source cannot accumulate stale markers. Repeated calls are stable and callers may mutate returned records. Ordinary rendering, `extract_translations`, and Babel translator comments continue independently.

Add `Environment.extract_marked_translations_from_templates(names=None)` to yield the same records for templates supplied by the configured loader. `names` is a finite iterable of logical template names, processed in its iteration order; when omitted, process all listed templates. Each record's `source` is the logical name being processed. This operation extracts source without rendering templates or recursively following includes or inheritance.
