"""Utilities for generating diffs between texts."""

from __future__ import annotations

from difflib import unified_diff
import importlib.util


def generate_unified_diff(original: str, corrected: str) -> str:
    """Generate a unified diff between original and corrected text.

    Args:
        original: Original text content
        corrected: Corrected text content

    Returns:
        String containing the unified diff
    """
    diff_lines = unified_diff(
        original.splitlines(),
        corrected.splitlines(),
        fromfile="original",
        tofile="corrected",
        lineterm="",
    )

    return "\n".join(diff_lines)


def generate_semantic_diff(original: str, corrected: str) -> str:
    """Generate a semantic diff between original and corrected text.

    Args:
        original: Original text content
        corrected: Corrected text content

    Returns:
        JSON string containing the semantic diff
    """
    import anyenv
    from diff_match_patch import diff_match_patch

    dmp = diff_match_patch()
    diffs = dmp.diff_main(original, corrected)
    dmp.diff_cleanupSemantic(diffs)

    # Convert to a JSON-friendly format
    result = []
    for op, text in diffs:
        op_name = {-1: "delete", 0: "equal", 1: "insert"}[op]
        result.append({"operation": op_name, "text": text})

    return anyenv.dump_json(result)


def generate_html_diff(original: str, corrected: str) -> str:
    """Generate an HTML visualization of the differences.

    Args:
        original: Original text content
        corrected: Corrected text content

    Returns:
        HTML string with highlighted differences
    """
    from diff_match_patch import diff_match_patch

    dmp = diff_match_patch()
    diffs = dmp.diff_main(original, corrected)
    dmp.diff_cleanupSemantic(diffs)

    return dmp.diff_prettyHtml(diffs)


def generate_all_diffs(original: str, corrected: str) -> dict[str, str]:
    """Generate all available diff formats between texts.

    Args:
        original: Original text content
        corrected: Corrected text content

    Returns:
        Dictionary with all available diff formats
    """
    result = {"unified_diff": generate_unified_diff(original, corrected)}

    if importlib.util.find_spec("diff_match_patch") is not None:
        result["semantic_diff"] = generate_semantic_diff(original, corrected)
        result["html_diff"] = generate_html_diff(original, corrected)

    return result
