# Show the chosen tip text in verbose proposal logs

When GroundedProposer uses random tip selection with verbose=True, print these
two adjacent lines once per proposal, using the selected key and its actual tip:

    Selected tip: <key>
    Chosen tip: <text> (tip_key=<key>)

The second line immediately follows the existing first line. Include the empty
text when the "none" key is selected. The logged tip must be the one passed to
the instruction generator. With verbose=False, print neither selection line;
with set_tip_randomly=False, do not fabricate a selection log. Preserve other
existing verbose output and proposal behavior.
