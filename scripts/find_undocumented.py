#!/usr/bin/env python
"""Find public functions and classes without docstrings.

Uses only the stdlib ``ast`` module — no extra dependencies required.

Example usage::

    python scripts/find_undocumented.py --path myosuite
    python scripts/find_undocumented.py --path myosuite --strict
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import sys
from pathlib import Path


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _has_docstring(node: ast.AST) -> bool:
    if not node.body:
        return False
    first = node.body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


def _find_undocumented(filepath: Path) -> list[tuple[int, str, str]]:
    """Return list of (line, kind, name) for public symbols missing docstrings."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if _is_public(node.name) and not _has_docstring(node):
                results.append((node.lineno, "class", node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_public(node.name) and not _has_docstring(node):
                results.append((node.lineno, "function", node.name))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find undocumented public symbols")
    parser.add_argument("--path", default="myosuite", help="Root path to search (default: myosuite)")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if any findings")
    parser.add_argument("--exclude", nargs="*", default=["*/tests/*", "*/agents/*"], help="Glob patterns to exclude")
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"Error: path {root} does not exist", file=sys.stderr)
        return 1

    total = 0
    for pyfile in sorted(root.rglob("*.py")):
        rel = str(pyfile)
        if any(fnmatch.fnmatch(rel, pat) for pat in args.exclude):
            continue

        findings = _find_undocumented(pyfile)
        for lineno, kind, name in findings:
            print(f"{pyfile}:{lineno}: {kind} {name}")
            total += 1

    print(f"\nTotal undocumented public symbols: {total}")
    if args.strict and total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
