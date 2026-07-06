# Report Folder Guide

This folder contains the write-up, figure assets, and the intermediate build artifacts used while generating the preprint.

## What To Read

- [mini_report.md](mini_report.md) is the main readable report.
- [2d_section.md](2d_section.md) explains the 2D GridWorld pipeline from data to figures.

## Figure Layout

- `figures/entropy/` contains the 1D information-theoretic figures.
- `figures/gridworld/` contains the 2D spatial-analysis figures.
- `figures/latent/`, `figures/jacobian/`, `figures/clustering/`, `figures/flow/`, and `figures/diffusion/` contain the earlier 1D figure sets.

## Build / Intermediate Artifacts

- `intermediate/` stores LaTeX conversions, logs, auxiliary files, and draft PDF builds.
- `tools/` contains small helper scripts used to convert or post-process the report.
- `intermediate/notes/` stores scratch notes that used to live at the top level.
- The top-level `report/` folder is kept for readable sources and convenience outputs; bulk build history lives in `intermediate/`.

## Suggested Reading Order

1. Read [mini_report.md](mini_report.md) for the main narrative.
2. Read [2d_section.md](2d_section.md) for the 2D workflow.
3. Inspect `figures/` for the generated outputs.
4. Use `intermediate/` only if you need the build history or conversion outputs.

## Notes

If you only want the clean reading path, start from `mini_report.md` and `2d_section.md`, then open the matching figure subfolders under `figures/`.