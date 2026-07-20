from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ALPHABET = tuple(
    "山水天地日月风云春秋江河花草松竹雨雪夜明青白远近高低"
    "东西南北人心梦归舟城门林石鸟星寒暖清幽"
)


def clustered_lines(count: int, variants: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    lines: list[str] = []
    used: set[str] = set()
    for index in range(count):
        cluster = index // variants
        variant = index % variants
        base_rng = random.Random(seed + cluster * 7919)
        characters = [base_rng.choice(ALPHABET) for _ in range(5)]
        position = variant % 5
        shift = 1 + variant // 5
        original = ALPHABET.index(characters[position])
        characters[position] = ALPHABET[(original + shift) % len(ALPHABET)]
        line = "".join(characters)
        while line in used:
            position = rng.randrange(5)
            characters[position] = rng.choice(ALPHABET)
            line = "".join(characters)
        used.add(line)
        lines.append(line)
    return lines


def generate_poems(count: int, shared_first_lines: int, seed: int) -> list[dict[str, object]]:
    if count < shared_first_lines or shared_first_lines <= 0:
        raise ValueError("count must be at least shared_first_lines > 0")
    first_pool = clustered_lines(shared_first_lines, variants=8, seed=seed)
    body_pool = clustered_lines(max(640, count // 2), variants=8, seed=seed + 1)
    poems: list[dict[str, object]] = []
    used_texts: set[str] = set()
    for index in range(count):
        group = index % shared_first_lines
        cycle = index // shared_first_lines
        offsets = (7, 19, 37)
        lines = [first_pool[group]] + [
            body_pool[(group * offset + cycle * (offset + 2) + index) % len(body_pool)]
            for offset in offsets
        ]
        text = "".join(lines)
        if text in used_texts:
            raise RuntimeError(f"Duplicate generated poem at index {index}")
        used_texts.add(text)
        poems.append(
            {
                "title": f"复合记忆试验{index:04d}",
                "author": "synthetic-stress",
                "lines": lines,
                "source_file": "generated",
                "source_id": f"stress-{index:04d}",
                "ambiguity_group": group,
            }
        )
    return poems


def write_dataset(path: Path, count: int, shared_first_lines: int, seed: int) -> None:
    poems = generate_poems(count, shared_first_lines, seed)
    group_counts = Counter(int(poem["ambiguity_group"]) for poem in poems)
    payload = {
        "description": "Synthetic high-conflict diagnostic for the Fig. 8(c) approximation.",
        "paper_dataset": False,
        "seed": seed,
        "stress_profile": {
            "poems": count,
            "shared_first_lines": shared_first_lines,
            "poems_per_first_line_min": min(group_counts.values()),
            "poems_per_first_line_max": max(group_counts.values()),
            "clustered_line_variants": 8,
            "alphabet_size": len(ALPHABET),
        },
        "poems": poems,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"saved {count} poems with {shared_first_lines} shared first-line groups "
        f"to {path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--shared-first-lines", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/fig8c_stress_poems.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_dataset(Path(args.output), args.count, args.shared_first_lines, args.seed)
