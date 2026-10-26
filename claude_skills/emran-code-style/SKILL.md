---
name: emran-code-style
description: >-
  Write Python in Emran's own coding style: long descriptive names, spaces
  around "=" in arguments, unaliased imports, dictionary-based configuration,
  numpy docstrings in a formal register, and traceable references under borrowed
  code. Use this skill whenever writing or refactoring Python for Emran's
  projects (reinforcement-learning agents, custom environments, Jupyter
  notebooks, matplotlib plotting), whenever Emran asks for code "in my style", "my
  format", "the way I code", or "match my conventions", whenever adding a
  function, class, or docstring to one of Emran's existing files so the new code
  reads as theirs, and whenever reformatting code to Emran's conventions. Apply it even
  when Emran does not say "my style" outright but is clearly asking for Python meant
  to live in their codebase. It governs code and its documentation, not report
  prose (that is the voice and academic writing skills).
---

# Writing in Emran's coding style

## What this is for

Emran writes Python with a consistent, deliberate style, and wants code written for them to match it on the first pass so it reads as their own and drops cleanly into their files. The domains are reinforcement-learning agents, custom gridworld environments, Jupyter notebooks, and matplotlib plotting, but the conventions are general Python.

This skill is about code and its documentation. Emran's report prose has its own skills (the narrative and academic writing components); when a docstring or comment contains prose, it follows Emran's formal register, described here and consistent with the academic component.

The single most important rule: **when editing an existing file, the surrounding code is the strongest guide — mirror its exact conventions.** The notes below are for new files, and for settling anything the surrounding code leaves open.

## The three lenses

Hold these together; go through them in order when checking code.

1. **Layout and naming.** The visual fingerprint: spaces around `=`, unaliased imports, long descriptive names, the whitespace and signature shape. Most of the "reads as Emran" signal lives here. Full detail in `references/style-guide.md`.
2. **Structure and configuration.** How Emran organises logic: dictionary-based configuration merged over defaults, a clear class lifecycle, base-class-plus-hooks subclassing, and reproducibility seeded throughout. Also in `references/style-guide.md`.
3. **Documentation.** numpy docstrings in Emran's formal register, comments that explain the *why*, and a traceable reference block beneath any borrowed code. Detail in `references/documentation.md`.

Read both reference files before writing more than a few lines; the conventions are specific and easy to miss.

## The fingerprint (hit these every time)

- **Spaces around `=` in keyword arguments and defaults** — `agent = Agent(environment = environment, seed = None)`, `def f(size: int, seed: int = None):`. Non-PEP8, and the clearest single tell of Emran's code.
- **No import aliases** — `import numpy` then `numpy.mean(...)`; `from matplotlib import pyplot` then `pyplot.subplots(...)`. Never `np` or `plt`.
- **Long, unabbreviated names**; `is_`/`has_` booleans; `__` for private methods and `_` for overridable hooks; `UPPER_CASE` constants; numeric literals grouped as `69_420`.
- **Multi-line signatures** — one parameter per line (past ~two params), indented two levels past the `def`, with `):` on its own line; a blank line right after the signature (and after the docstring) before the body.
- **Generous vertical whitespace** — two, sometimes three, blank lines between methods.
- **Dictionary configuration** merged over defaults (`self.parameters.update(passed_in)`); everything a caller might reasonably change is configurable, not hard-coded.
- **numpy docstrings** in a formal, impersonal register; **why-comments**; and `# Reference` / `# Source:` / `# Link:` beneath borrowed code.
- **British spelling** in docstrings and comments; code identifiers follow the library.

## How to work

Understand what the code should do before styling it; the style is presentation, and a correct, clear solution comes first.

When extending an existing file, read it and copy its conventions exactly — that is the whole point. When starting fresh, apply the full style from the references. Keep things configurable rather than rigid; Emran has asked for this specifically, preferring a parameter passed through a dictionary over a hard-coded constant.

Then reread with fresh eyes against the three lenses and the checklist below.

## Checklist before handing code over

- Spaces around `=` in every keyword argument and default value.
- Imports unaliased (`numpy`, `pyplot` spelled out), grouped stdlib / third-party / local.
- Names long and descriptive; booleans as `is_`/`has_`; `__` private and `_` hooks; `UPPER_CASE` constants; `69_420`-style literals.
- Signatures with more than a couple of parameters split one per line, with `):` on its own line and a blank line after.
- Two blank lines between methods; blank lines grouping logical blocks.
- Configuration passed as a dictionary and merged over defaults; nothing needlessly hard-coded.
- Type hints on signatures, including return types.
- numpy docstrings in the formal register; comments explain the why; borrowed code carries a real, traceable `# Reference` block.
- British spelling in prose; library identifiers left as they are.
