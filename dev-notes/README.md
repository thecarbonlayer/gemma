# dev-notes

Design decisions and the evolution backlog for gemma's library surface.

The fifteen-chapter curriculum (`ch-00`..`ch-14`) is the finished teaching
build. This directory tracks what happens after it: the embedding seam and the
editable surface that external consumers and the self-improving loop keep
pushing on.

- [adr/](adr/) — architecture decision records.
  - [0001](adr/0001-version-the-evolution-separately.md) — version the evolution on its own axis, not as new chapters.
  - [0002](adr/0002-mechanism-in-gemma-domain-in-the-consumer.md) — generic mechanism in gemma, domain and policy in the consumer.
- [sdk-seam-roadmap.md](sdk-seam-roadmap.md) — the prioritized backlog for the embedding seam, grounded in real consumers.
