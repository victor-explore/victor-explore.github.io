# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hugo static site blog (victor-explore.github.io) using the PaperMod theme. The site focuses on AI/ML topics including deep learning, reinforcement learning, and technical explanations.

## Commands

```bash
# Run local development server (includes draft posts)
hugo server -D

# Run with full recompilation on changes (slower but more reliable)
hugo server --disableFastRender

# Build for production
hugo --minify
```

## Deployment

Automatic via GitHub Actions on push to `main` branch. The workflow builds with `hugo --minify` and deploys to GitHub Pages.

## Content Structure

- All posts go in `content/posts/` organized by topic folders (ADRL, Books, n8n, etc.)
- Posts use YAML front matter with: title, date, draft, description, tags, categories, weight
- Use `weight` in front matter to control ordering within a section
- Internal links use Hugo shortcodes: `{{< ref "/posts/folder/file.md" >}}` or `{{< relref "..." >}}`

## Writing Guidelines

When writing technical/mathematical content:

- Use `$` for inline math, `$$` for block equations
- Explain equations progressively: first explain inner terms using underbraces, then build up to the full equation
- Example equation annotation pattern:
```latex
$$
\underbrace{\mu_k^*(x_k)}_{\substack{\text{Optimal policy} \\ \text{at stage } k}} = \arg\min_{a_k} \underbrace{E[\ldots]}_{\substack{\text{Expected cost}}}
$$
```
- Use bullet points and headings for readability
- Build intuition for why concepts matter and how they fit the bigger picture
- Maintain logical flow between concepts

## Configuration

Main config in `hugo.toml`:
- Math rendering enabled (`math = true`)
- Table of contents enabled by default
- Unsafe HTML allowed in markdown
