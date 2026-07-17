# Changelog

The teaching curriculum is versioned by chapter tags (`ch-00`..`ch-14`) and does
not change. This file tracks the other axis: the `gemma` library and its
editable surface, which evolve continuously as external consumers and the
self-improving loop push on the seam. See
[dev-notes/adr/0001](dev-notes/adr/0001-version-the-evolution-separately.md).

The format follows [Keep a Changelog](https://keepachangelog.com/), and versions
follow [Semantic Versioning](https://semver.org/). The configuration's own
integer `version` field is the fine-grained counter underneath these releases.
One entry per release; commits stay fine-grained under a `feat(surface)` or
`feat(sdk)` scope.

## [Unreleased]

The first release, `v0.1.0`, opens the embedding seam: the surface external code
uses to build domain-specific agents on the harness. Planned contents, tracked
in [dev-notes/sdk-seam-roadmap.md](dev-notes/sdk-seam-roadmap.md):

### Added

- Structured run result from `Agent.run` (final text, tool calls, totals, turns, approvals, stop reason). `Agent.send` keeps returning the final text.
- Schema-constrained output mode on `chat()` and `Agent`.
- Public `ToolRegistry` introspection (get/wrap/override/list), a per-call attribute bag and `is_error` flag, and a `subscribe()` event stream.
- `Provider.from_env(root=)`, a public `load_env()`, model params as agent config, and a `provenance()` stamp.
- `config_schema()` introspection alongside the public `load_config` door.
- A curated, semantically versioned `gemma` package. Existing module paths keep working.

### Changed

- The approval gate consults a `Policy` object (allow, deny, read-only, path scope, predicate) instead of a global tool-name set plus a yes/no callback. Existing constructor arguments keep working through a compatibility layer.
