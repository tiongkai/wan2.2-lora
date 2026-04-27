---
name: llm-wiki
description: Use when building or maintaining a persistent LLM-managed knowledge base or wiki; when ingesting sources into a structured wiki, querying accumulated knowledge, running wiki health checks, or designing a personal/team knowledge system where knowledge should compound across sessions rather than be re-derived each time.
---

# LLM Wiki

## Overview

**Stop re-deriving — start compiling.** An LLM-maintained wiki is a persistent, compounding artifact: cross-references already built, contradictions already flagged, synthesis already current. RAG retrieves and forgets. A wiki accumulates and grows.

## Architecture (3 Layers)

| Layer | What it is | Who owns it |
|-------|-----------|-------------|
| **Raw sources** | Immutable input documents (articles, papers, transcripts) | Human |
| **Wiki** | LLM-generated markdown pages: summaries, entities, concepts, index | LLM |
| **Schema** | CLAUDE.md / AGENTS.md — conventions, ingest/query/lint workflows, domain types | Co-evolved |

## Core Operations

**Ingest** — Drop a source, tell the LLM to process it. LLM reads → discusses key takeaways → writes summary page → updates index → updates entity/concept pages → appends to log. One source typically touches 10–15 wiki pages.

**Query** — Ask a question. LLM reads `index.md` → drills into relevant pages → synthesizes answer with citations. Good answers get filed back as new wiki pages — explorations compound just like ingested sources.

**Lint** — Periodic health check. Find: contradictions, stale claims, orphan pages, missing cross-references, concepts without a page. LLM suggests new sources and questions.

## Index and Log

- **`index.md`** — content catalog: every page with link, one-line summary, metadata. LLM reads this first on every query. Works well up to ~100–200 pages.
- **`log.md`** — append-only chronological record. Use consistent prefix `## [YYYY-MM-DD] operation | title` so it's grep-parseable.

## Quick Reference: Page Types

| Type | Purpose |
|------|---------|
| Entity page | Named thing (person, project, library) with attributes and relationships |
| Concept page | Topic, idea, or technique |
| Source summary | Key takeaways from one raw source |
| Synthesis page | Cross-source analysis or comparison |
| Overview | Entry point, orientation to the whole wiki |

## V2: Production Patterns

When the wiki grows past ~200 pages or needs to stay healthy long-term:

**Memory lifecycle** — Add confidence scores to facts (source count, recency, contradictions). Implement supersession: new info explicitly replaces old with a timestamp link. Apply retention decay — architecture decisions decay slowly, transient bugs decay fast.

**Consolidation tiers** (working → episodic → semantic → procedural): promote facts up as evidence accumulates; raw observations eventually become established patterns.

**Knowledge graph** — Extract typed entities and relationships (`uses`, `depends on`, `caused`, `supersedes`). Graph traversal catches connections keyword search misses. Pages are for reading; the graph is for navigation.

**Hybrid search** — Past ~200 pages replace index-only search with BM25 + vector + graph traversal, fused with reciprocal rank fusion. [qmd](https://github.com/tobi/qmd) is a ready-made option (CLI + MCP server).

**Automation hooks** — Event-driven: on new source (auto-ingest, entity extraction, index update); on session start (inject relevant context); on session end (compress into observations); on schedule (lint, retention decay, consolidation).

**Quality** — Score LLM-generated content on structure, citation, consistency. Self-healing lint: auto-fix orphans, stale claims, broken cross-references. Contradiction resolution: LLM proposes which claim wins based on recency and source authority.

**Crystallization** — After a completed research thread or debugging session, auto-distill it into a structured digest (question, findings, entities, lessons). File as a first-class wiki page.

## Implementation Spectrum

Start minimal, add layers as needed:

1. **Minimal**: raw sources + wiki pages + `index.md` + schema
2. **+ Lifecycle**: confidence scoring, supersession, basic retention decay
3. **+ Structure**: entity extraction, typed relationships, knowledge graph
4. **+ Automation**: hooks for auto-ingest, auto-lint, context injection
5. **+ Scale**: hybrid search, consolidation tiers, quality scoring
6. **+ Collaboration**: mesh sync, shared/private scoping, multi-agent coordination

## Tooling Tips

- **Obsidian** as the wiki IDE — graph view shows hubs and orphans; Web Clipper converts articles to markdown
- **Dataview plugin** — query YAML frontmatter for dynamic tables across pages
- **Marp** — generate slide decks from wiki content
- **Download images locally** — LLMs can't read inline images in one pass; download assets to `raw/assets/` and reference them separately
- The wiki is a git repo — version history and branching come free

## The Schema Is the Real Product

The schema (CLAUDE.md/AGENTS.md) is the most important file. It encodes entity/relationship types, ingest conventions, quality standards, contradiction handling, consolidation schedule, and private-vs-shared scoping. Co-evolve it with the LLM over time. A mature schema is transferable to anyone working in the same domain.
