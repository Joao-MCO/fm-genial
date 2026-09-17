from __future__ import annotations

import os
from paths import BASE_DIR, DATABASES_DIR


def resolve_csv_path(filename: str) -> str:
    filename = str(filename)
    names = [filename]
    if not filename.lower().endswith(".csv"):
        names.append(f"{filename}.csv")

    candidates = []
    for name in names:
        candidates.extend([
            name,
            os.path.join(BASE_DIR, name),
            os.path.join(DATABASES_DIR, name),
        ])
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Arquivo não encontrado: " + filename + "\n" + "\n".join(candidates))


def resolve_attributes_path(stats_path: str, attributes_path: str | None = None) -> str:
    if attributes_path:
        return resolve_csv_path(attributes_path)
    return resolve_csv_path(os.path.join("attributes", os.path.basename(stats_path)))