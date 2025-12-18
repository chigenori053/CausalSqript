"""Minimal `yaml` module shim for environments without PyYAML."""

from __future__ import annotations

from coherent_yaml import dump, safe_load

FullLoader = object  # compatibility placeholder
__all__ = ["safe_load", "dump", "FullLoader"]
