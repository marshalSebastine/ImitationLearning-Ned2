---
name: coding-style-emran-code-style
description: "Emranul's own Python coding style lives in the emran-code-style skill; load it for any Python written for them"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dd705f28-62f1-4a95-afe4-b6bd1767c0dc
  modified: 2026-10-23T16:36:05.947Z
---

Emranul's coding style is packaged as the user-level skill `emran-code-style` (tracked in the repo at `claude_skills/emran-code-style`, linked into Claude by `setup_claude.sh` or `setup_claude.ps1`). It was installed on 2026-10-23 and renamed from `afraaz-code` at Emranul's request. It calls for:
- spaces around `=` in arguments and defaults;
- unaliased imports (`numpy`, `pyplot`);
- long descriptive names;
- dictionary configuration merged over defaults;
- numpy docstrings with British spelling;
- `# Reference` blocks under borrowed code.

In their own words, they want code that is "really structured, mini, well organised, as less code as possible".

**Why:** they want code that reads as their own and drops into their files unchanged, for their master's module project in `AI_Project` (the shared GitHub repo) (started 2026-10-23).

**How to apply:** load the `emran-code-style` skill before writing Python for them. Keep the logic lean (no duplication, no speculative features), but still follow the skill's layout rules. Where the two pull apart (the skill's generous whitespace and full docstrings against "less code"), follow the skill's layout and keep the line count down through simpler logic.
