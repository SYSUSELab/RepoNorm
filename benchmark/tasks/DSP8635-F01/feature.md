# Proposal cache salts without temperature perturbation

Stop perturbing the prompt model's temperature to distinguish instruction
proposals. Keep the T argument for call compatibility, but do not mutate prompt
model kwargs while proposing, including when generation raises an exception.

Give each proposal a nonempty ASCII cache_salt and pass the same value to its
DescribeProgram, DescribeModule and GenerateSingleModuleInstruction predictor
calls when those stages are enabled. Their signatures expose cache_salt as an
input with prefix "CACHE SALT (IGNORE):". Separate proposals must receive distinct
salts, while identical seeded RNG state and proposal call sequence reproduce the
same salt sequence. Preserve program-aware and dataset-summary switches.

Direct GenerateModuleInstruction calls without an explicit salt must also obtain
one shared value for their enabled stages; an explicitly supplied value is used
unchanged. Support RNG objects providing the existing random/choice/uniform
interface without requiring getrandbits. No particular private helper, alphabet
or salt-generation algorithm is required.
