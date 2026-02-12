# Copilot Commit Instructions

Write a Git commit message that summarizes **only the currently staged changes**.

## Format (required)

1. Start with **2–4 short lines** that describe the overall intent and context of the staged change set (what/why). Keep it plain text.
2. Add **one blank line**.
3. Add a **numbered list** describing the staged changes.

## Numbered list rules (required)

- Include **every staged file exactly once** in the list (no duplicates, no omissions).
- Each item must be: `N. **path/to/file.ext**: <single concise change description>`
- Use **present tense**, be specific, and avoid vague verbs (e.g., "update", "fix stuff").
- If multiple changes exist in one file, summarize them as one combined description.

## Content constraints

- **Do not** mention unstaged/untracked files.
- **Do not** include commands, hashes, PR links, or speculation.
- **Do not** add any extra sections (no "Notes", "Testing", etc.) unless the staged diff clearly includes such changes.
- Output **only** the commit message text (no code fences, no commentary).

## Example

Improve documentation build configuration.

1. **docs/mkdocs.yml**: Enable autorefs and adjust theme extensions.
2. **pyproject.toml**: Add documentation dependencies to the doc group.