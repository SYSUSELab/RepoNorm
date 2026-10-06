**Title:** Track application translation versions per Environment

The i18n extension adds `set_translation_version(message, version)`, `get_translation_version(message)`, and `compare_translation_versions(message1, message2)`. Messages and versions are strings. Setting updates the exact message; getting an unknown message returns `None`. Comparison is true only when both messages are registered and their versions are equal, so either or both missing always returns false.

`remove_translation_version(message)` removes one entry and is harmless for an unknown message. `clear_translation_versions()` removes every entry. Empty, Unicode, and long strings are supported. The storage representation and hashing are unspecified.

Each Environment has independent state. An overlay begins with a copy of its parent's versions and subsequent updates and removals are isolated in both directions. Version tracking does not change installation, extraction, or rendering of translations.
