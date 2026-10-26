# Documentation: docstrings, comments, and references

Three conventions cover how Emran documents code: numpy docstrings, why-comments, and a traceable reference block under borrowed code.

## Docstrings

- **numpy style.** A one-line summary, an optional short elaboration, then `Parameters` and `Returns` sections, and `Raises` where a function raises. Use the `----------` underlines under each section heading.
- **The prose follows Emran's formal register** — impersonal, precise, and plain, the same voice as Emran's report writing. Describe what the function does and why; avoid casual asides ("because Emran likes the dark one", "a sneaky peek", "if it gets lucky") and figurative language. British spelling.
- **The summary line is imperative and concrete**: "Score an archived state as a candidate to return to and expand.", "Return whether a cell is water at a given step.", "Manhattan distance between two observations, measured from the start."
- Reference identifiers in double backticks: ``reset``, ``self.goal_trajectory``, ``(row, column)``.

Example:

```python
def __calculate_frontier_score(self, state: Tuple[int, int]):
    """
    Score an archived state as a candidate to return to and expand.

    The score favours a state whose best untried action leads somewhere novel
    and distant from the start, and penalises states that are heavily visited
    or frequently chosen.

    Parameters
    ----------
    state : tuple of int
        The ``(observation, time)`` archive key being scored.

    Returns
    -------
    float or None
        The frontier score, or ``None`` if the state has no untried, non-lethal
        action remaining.
    """
```

## Comments

- **Explain the why, not the what.** A comment earns its place by giving the reasoning the code cannot: "the wave repeats only after an irrational number of steps, so the phase axis is a rounded integer approximation of that true period." A comment that merely restates the next line is noise.
- Lowercase, sitting on the line or lines directly above the code they explain.
- Keep them clean and impersonal in submitted work; the casual shorthand Emran sometimes uses in drafts (for example "cz" for "because") does not belong in code that is handed in.

## References for borrowed code

When a block of code follows a published source, put a reference directly beneath it. This keeps the work traceable and is what protects it from being read as plagiarism. The exact format:

```python
# Reference
# Source: <who or what> - <one line describing the borrowed idea or snippet>.
# Link: <url that actually contains that code or a close resemblance>
```

A worked example:

```python
row = observation // self.column_pool
column = observation % self.column_pool

# Reference
# Source: Gymnasium FrozenLake - row-major flat-index encode/decode
#         (to_s = row * ncol + col; row = s // ncol; col = s % ncol).
# Link: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py
```

Rules that matter:

- **The link must genuinely contain the code, or a close resemblance.** A reference that points somewhere the marker cannot find the resembling code is worse than none; it reads as a fabricated citation.
- **Never cite an AI assistant, a chatbot, or a prompt as the source.** Cite the real published origin: official documentation, a paper, a public repository, or a tutorial.
- **Only borrowed code needs a reference.** Original logic stands on its own; do not attach a citation to your own work.
