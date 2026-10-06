**Title**: Add Token Compression
**Pull Request Details**
**Description**:
Add token compression functionality that reduces a returned token sequence by collapsing adjacent repeated token IDs

**Technical Background**:
**Problem**: Some applications need a shortened sequence representation that collapses adjacent equal token IDs. This is a lossy sequence transformation: it does not preserve the original text or guarantee unchanged meaning, and is not a reversible compression format.

Example of the issue:
```python
# Encode text with repetitive content
text = "The dog dog dog ran ran quickly"
encoded = enc.encode(text)  # Results in more tokens than necessary
# No way to compress repeated adjacent tokens like "dog dog dog" -> "dog"
```

Add an optional `compress` parameter to `encode`. When true, each consecutive run of the same token ID must be represented by one token ID in the returned sequence; repeated IDs separated by another token remain distinct. When false or omitted, existing encoding behavior must be preserved.
