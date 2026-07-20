from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def split_clauses(paragraphs: list[str]) -> list[str]:
    clauses: list[str] = []
    for paragraph in paragraphs:
        current: list[str] = []
        for character in paragraph:
            if unicodedata.category(character).startswith("P") or character.isspace():
                if current:
                    clauses.append("".join(current))
                    current = []
            else:
                current.append(character)
        if current:
            clauses.append("".join(current))
    return clauses


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_dir: Path, output: Path, count: int) -> None:
    sources = sorted(
        source_dir.glob("poet.tang.*.json"),
        key=lambda path: int(path.stem.rsplit(".", 1)[-1]),
    )
    poems: list[dict[str, object]] = []
    seen_titles: set[str] = set()
    seen_texts: set[str] = set()

    for source in sources:
        entries = json.loads(source.read_text(encoding="utf-8"))
        for entry in entries:
            title = str(entry.get("title", "")).strip()
            lines = split_clauses(list(entry.get("paragraphs", [])))
            text = "".join(lines)
            if (
                not title
                or title in seen_titles
                or text in seen_texts
                or len(lines) != 4
                or any(len(line) != 5 for line in lines)
            ):
                continue
            poems.append(
                {
                    "title": title,
                    "author": str(entry.get("author", "")).strip(),
                    "lines": lines,
                    "source_file": source.name,
                    "source_id": str(entry.get("id", "")),
                }
            )
            seen_titles.add(title)
            seen_texts.add(text)
            if len(poems) == count:
                break
        if len(poems) == count:
            break

    if len(poems) < count:
        raise ValueError(f"Only {len(poems)} eligible poems found; requested {count}.")

    payload = {
        "description": "Deterministic nonauthor sample for the Fig. 8(c) approximation.",
        "selection": "First unique-title/text poems with four five-character clauses.",
        "source_repository": "https://github.com/chinese-poetry/chinese-poetry",
        "source_files": [
            {"name": source.name, "sha256": sha256(source)} for source in sources
        ],
        "poems": poems,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(poems)} poems to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/chinese_poetry_source")
    parser.add_argument("--output", default="data/fig8c_poems.json")
    parser.add_argument("--count", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(Path(args.source_dir), Path(args.output), args.count)
