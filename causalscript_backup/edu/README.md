# Edu Directory Overview

This tree follows the requirements from `docs/MathLang UI Directory Specification.md`:

- `ui/` — rendering helpers and layout primitives used by notebooks/CLI.
- `notebooks/` — starter notebooks for demos (placeholders tracked via README).
- `lessons/` — YAML/JSON lesson definitions consumable by CLI/Notebook workflows.
- `demo/` — CLI runner wrapper (`edu_demo_runner.py`) and scenario docs/config.
- `config/` — UI settings (`edu_ui_settings.yaml`, themes, demo config).
- `examples/` — DSL programs in v2.5 format that power demos/tests.

Each subdirectory now contains at least one file so contributors can expand it without recreating structure.
