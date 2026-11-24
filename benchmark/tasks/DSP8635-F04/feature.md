# Limit generated program and module descriptions

Add max_description_chars=2000 to GroundedProposer and pass the setting through
to description use during instruction generation. None disables the limit;
otherwise accept only positive, non-boolean integers and reject invalid values
with ValueError.

Before a generated description is used in a later predictor prompt, ensure its
final string length is no greater than the cap. Apply this to both program and
module descriptions, including the program description passed to DescribeModule.
Descriptions already within the cap are unchanged.

For an overlong description, reserve one character for the single-character
ellipsis … and retain the last complete sentence ending (. ! or ?, followed by
whitespace or the end) that fits. If none fits, take a hard prefix of cap-1
characters. Remove trailing whitespace from that prefix and append …. Count
Python string characters, not encoded bytes. A cap of one produces only ….
The None setting and program-aware disabled path retain existing behavior.
