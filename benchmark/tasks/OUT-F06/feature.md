**Title**: Add inheritable grammar validation policy to XGrammar

Allow `XGrammarBackend(model, validate_grammar=None)` to configure a default validation callback. The JSON-schema, regex, and CFG processor factories also accept `validate_grammar`.

If a factory omits this argument, use the backend default. An explicit callback overrides the default for that request; explicit `None` disables validation for that request. Selection must not alter the backend default or affect another backend instance.

The selected callback runs exactly once on the original grammar string before compilation. Any normal return value (including `False`) means validation completed. If it raises, propagate that same exception without compiling or creating a processor for this request; subsequent requests must remain usable. With no selected callback, compile normally. Preserve original factory and processor behavior when no validation is configured.
