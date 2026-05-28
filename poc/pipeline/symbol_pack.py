"""Resolves which symbol-template PNGs to feed to `cv_detect.detect_symbols`
based on a `symbol_pack` selector ("builtin" / "user" / "both").

Built-in templates live at `symbol_library/*.png` (one per class, top-level).
User templates live at `symbol_library/user/*.png` (one per class, ingested
from a legend page).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .rag import USER_SYMBOL_DIR, _SYMBOL_LIBRARY_DIR


SymbolPack = Literal["builtin", "user", "both"]


def resolve_symbol_pngs(
    pack: SymbolPack,
    builtin_dir: Path = _SYMBOL_LIBRARY_DIR,
    user_dir: Path = USER_SYMBOL_DIR,
) -> list[Path]:
    """Return the list of PNGs the CV detector should use for `pack`.

    Resolution rules:
      - "builtin": only top-level PNGs in builtin_dir
      - "user":    only PNGs in user_dir
      - "both":    union, with user/* shadowing same-stem builtin/* files
    """
    if pack == "builtin":
        return sorted(builtin_dir.glob("*.png"))
    if pack == "user":
        if not user_dir.exists():
            return []
        return sorted(user_dir.glob("*.png"))
    if pack == "both":
        merged: dict[str, Path] = {}
        for p in sorted(builtin_dir.glob("*.png")):
            merged[p.stem] = p
        if user_dir.exists():
            for p in sorted(user_dir.glob("*.png")):
                merged[p.stem] = p   # user shadows builtin
        return [merged[k] for k in sorted(merged)]
    raise ValueError(f"Unknown symbol pack: {pack!r}")


def user_pack_size(user_dir: Path = USER_SYMBOL_DIR) -> int:
    if not user_dir.exists():
        return 0
    return sum(1 for _ in user_dir.glob("*.png"))
