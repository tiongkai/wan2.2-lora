---
name: portable-memory-in-repo
description: Keep session memory mirrored into the wan2.2-lora repo so work continues across machines
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23bc0f31-dbdd-4026-950d-7fdbb13e8808
---

The user wants project memory/state logged INTO the wan2.2-lora repo (not just `~/.claude`) so they can continue on other machines.

**Why:** `~/.claude/projects/.../memory/` is machine-local and doesn't travel. The repo has a GitHub remote (`origin → github.com/tiongkai/wan2.2-lora.git`), so committing docs makes them portable.

**How to apply:** Keep `wan2.2-lora/HANDOFF.md` (portable project state — current artifacts, gotchas, run commands, next steps) and `wan2.2-lora/docs/session-memory/` (mirror of the `~/.claude` memory `.md` files) up to date as work progresses. Large binaries (`models/`, `loras/`, `logs/`, `*.safetensors`, `*.pth`) are gitignored — they must be transferred separately, NOT committed (50 GB+). Offer to commit+push when meaningful state changes; the user is on the `main` branch. See [[comfyui-generation-setup]] [[i2v-lora-training]].
