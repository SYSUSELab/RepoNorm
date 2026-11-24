# Third-party provenance

RepoNorm retains upstream notices and identifies the imported components here.

| Component | Source revision | Included use and adaptations |
| --- | --- | --- |
| [CooperBench](https://github.com/cooperbench/CooperBench) | `4913c4ebb84d2606cdb5628936b88529f3e181df` | Selected mini-SWE-agent v2 runtime, communication helpers, and test-file filtering; lightweight imports and public packaging |
| [CodeWiki](https://github.com/FSoft-AI4Code/CodeWiki) | `f5dac8c42096af4242f2fac283b4bde1e36ba065` | Documentation generator and tests; snapshot selection, provider instrumentation, isolated execution, and static export adapter |
| [coding-agent-wrapper](https://github.com/anhnh2002/caw) | `9ec22af22a32ee50392443ae00ca401c3ec4e46b` | Pinned dependency of the recorded CodeWiki environment |

The imported CooperBench and CodeWiki package metadata declare MIT licensing.
The acquisition package and experiment adapters are maintained in this release.
The acquisition discovery runner uses mini-SWE-agent 2.4.6; the coding experiments
use the separate CooperBench mini-SWE-agent v2 adapter. They are distinct runtime
components.

Benchmark source snapshots remain in their upstream repositories. Their URLs,
commits, and image identifiers are recorded in `benchmark/manifest.json`.
Preparation clones them into ignored output directories. Task and patch data
retain the selected experiment bytes and their source hashes.
