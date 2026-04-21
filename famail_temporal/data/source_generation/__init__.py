"""Unified GPS source-data generation tool.

Single-entry-point tool that reads the 3 raw taxi GPS pickle files and
produces all 8 source datasets consumed by `famail_temporal`. Cross-file
consistency is enforced by construction: everything derives from one
enriched event stream produced in one pass.

See docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md
for the full design rationale.
"""
