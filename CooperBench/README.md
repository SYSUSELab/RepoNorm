# CooperBench runtime

Selected runtime components from [CooperBench](https://github.com/cooperbench/CooperBench),
commit `4913c4ebb84d2606cdb5628936b88529f3e181df`, including its vendored
mini-SWE-agent v2 adapter. The source metadata declares the MIT license.

This subset supports the single-agent Docker experiments through `runner/`.
Package initializers are lightweight; unrelated agent integrations and benchmark
data are omitted. The agent implementation and prompt templates are retained.
