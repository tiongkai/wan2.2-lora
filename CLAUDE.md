# Wan 2.2 LoRA Training Wiki — Schema

## Purpose

A compounding knowledge base on training LoRA adapters for the Wan 2.1/2.2 video generation model family. The wiki captures techniques, tool configurations, dataset preparation practices, and community-validated settings so knowledge compounds across sessions rather than being re-derived.

## Directory Structure

```
raw/              # Immutable source documents (human-owned)
  articles/       # Web articles saved as markdown
  videos/         # YouTube video metadata + transcript notes
  papers/         # Academic papers and technical docs
  assets/         # Downloaded images, diagrams, screenshots
wiki/             # LLM-generated pages (LLM-owned)
  overview.md     # Entry point — orientation to the whole wiki
  index.md        # Content catalog — read this first on every query
  log.md          # Append-only chronological record
  concepts/       # Topic, idea, or technique pages
  entities/       # Named things (tools, models, libraries)
  sources/        # Key takeaways from one raw source
  synthesis/      # Cross-source analysis or comparison
```

## Entity Types

| Type | Examples | Key Attributes |
|------|----------|---------------|
| Model | Wan 2.1, Wan 2.2 | architecture, parameters, variants, release date |
| Tool | musubi-tuner, AI Toolkit, diffusion-pipe | repo URL, supported models, key features, install method |
| Training Concept | LoRA rank, learning rate, overfitting | definition, recommended values, tradeoffs |
| Hardware | RTX 4090, A100, H100 | VRAM, typical training time, cost tier |

## Relationship Types

- `trains` — Tool → Model (e.g., musubi-tuner trains Wan 2.2)
- `configures` — Parameter → Training run
- `requires` — Model variant → minimum VRAM
- `supersedes` — Wan 2.2 supersedes Wan 2.1
- `recommended-for` — Setting → Use case (e.g., rank 16 recommended-for identity LoRA)
- `sources` — Claim → Source page

## Page Conventions

### Frontmatter
Every wiki page has YAML frontmatter:
```yaml
---
title: Page Title
type: concept | entity | source | synthesis | overview
tags: [relevant, tags]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### Cross-References
Use `[[page-name]]` wiki-links to reference other pages. Link liberally.

### Source Citations
Cite sources inline as `[source-short-name]` linking to the source summary page.

### Trigger Tokens
When documenting trigger tokens for LoRA training, always use code formatting: `tok_person`, `sks_style`.

## Ingest Workflow

1. Save raw source to `raw/` (article markdown, video metadata, paper PDF)
2. Read the raw source
3. Write a source summary page in `wiki/sources/`
4. Update or create entity pages in `wiki/entities/`
5. Update or create concept pages in `wiki/concepts/`
6. Update `wiki/index.md`
7. Append to `wiki/log.md`

## Quality Standards

- Every claim should trace to a source page
- Flag contradictions between sources explicitly
- Mark community-consensus settings vs. individual experiments
- Note hardware-specific caveats (what works on 3090 may differ on A100)
- Prefer specific numbers over vague guidance
- Date-stamp time-sensitive information (tool versions, model releases)
