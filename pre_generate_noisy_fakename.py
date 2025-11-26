"""
Utility script to create noisier FakeName datasets.

Example:
python pre_generate_noisy_fakename.py --noise-level 0.8 --source-dir data/datasets/synthetic/ --output-dir data/datasets/noisy
"""
import argparse
import csv
import pathlib
import random
import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional


def clamp(prob: float, maximum: float = 0.95) -> float:
    return max(0.0, min(prob, maximum))


def build_noise_config(level: float) -> Dict[str, float]:
    # Probabilities scale with the requested level; capped to keep the data usable.
    return {
        "missing_prob": clamp(0.03 * level),
        "typo_prob": clamp(0.15 * level),
        "case_prob": clamp(0.1 * level),
        "swap_name_prob": clamp(0.04 * level),
        "whitespace_prob": clamp(0.12 * level),
        "suffix_prob": clamp(0.05 * level),
        "uid_noise_prob": clamp(0.06 * level),
        "date_shift_prob": clamp(0.3 * level),
        "date_format_prob": clamp(0.45 * level),
        "date_text_token_prob": clamp(0.02 * level),
        "max_date_shift_days": max(1, int(12 * level)),
    }


def introduce_typo(value: str, rng: random.Random) -> str:
    if not value:
        return value
    idx = rng.randrange(len(value))
    operations = ("delete", "insert", "swap", "replace")
    op = rng.choice(operations)
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op == "delete":
        return value[:idx] + value[idx + 1 :]
    if op == "insert":
        return value[:idx] + rng.choice(letters) + value[idx:]
    if op == "swap" and len(value) > 1:
        j = min(idx + 1, len(value) - 1)
        swapped = list(value)
        swapped[idx], swapped[j] = swapped[j], swapped[idx]
        return "".join(swapped)
    return value[:idx] + rng.choice(letters) + value[idx + 1 :]


def random_case(value: str, rng: random.Random) -> str:
    if not value:
        return value
    fn = rng.choice([str.lower, str.upper, str.title, str.capitalize])
    return fn(value)


def add_whitespace(value: str, rng: random.Random) -> str:
    if not value:
        return value
    prefix = " " * rng.randint(0, 2)
    suffix = " " * rng.randint(0, 2)
    return f"{prefix}{value}{suffix}"


def add_suffix(value: str, rng: random.Random) -> str:
    if not value:
        return value
    suffixes = [" Jr", " Sr", " II", " III", "-Smith"]
    return value + rng.choice(suffixes)


def parse_birthday(value: str) -> Optional[datetime]:
    # Keep formats portable (no %-m, which breaks on Windows).
    formats = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def format_birthday(dt: datetime, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["date_text_token_prob"]:
        return rng.choice(["unknown", "n/a", "see notes", "??"])
    if rng.random() < config["date_shift_prob"]:
        delta = rng.randint(-config["max_date_shift_days"], config["max_date_shift_days"])
        dt = dt + timedelta(days=delta)
    formats = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d %b %Y"]
    formatted = dt.strftime(rng.choice(formats))
    if rng.random() < config["date_format_prob"]:
        formatted = re.sub(r"\b0(\d)", r"\1", formatted)
        if rng.random() < 0.3:
            formatted = formatted.replace("/", "-")
    return formatted


def corrupt_uid(uid: str, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["missing_prob"]:
        return ""
    uid = uid.strip()
    if not uid:
        return uid
    if rng.random() > config["uid_noise_prob"]:
        return uid
    ops = ("drop", "swap", "pad", "space")
    op = rng.choice(ops)
    if op == "drop" and len(uid) > 1:
        idx = rng.randrange(len(uid))
        return uid[:idx] + uid[idx + 1 :]
    if op == "swap" and len(uid) > 1:
        idx = rng.randrange(len(uid) - 1)
        swapped = list(uid)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        return "".join(swapped)
    if op == "pad":
        return uid + rng.choice(["0", "00", " "])
    return f"{uid[:2]} {uid[2:]}"


def mutate_name(value: str, rng: random.Random, config: Dict[str, float]) -> str:
    if rng.random() < config["missing_prob"]:
        return ""
    if rng.random() < config["typo_prob"]:
        value = introduce_typo(value, rng)
    if rng.random() < config["case_prob"]:
        value = random_case(value, rng)
    if rng.random() < config["whitespace_prob"]:
        value = add_whitespace(value, rng)
    if rng.random() < config["suffix_prob"]:
        value = add_suffix(value, rng)
    return value


def mutate_row(row: Dict[str, str], rng: random.Random, config: Dict[str, float]) -> Dict[str, str]:
    mutated = dict(row)
    mutated["GivenName"] = mutate_name(row.get("GivenName", ""), rng, config)
    mutated["Surname"] = mutate_name(row.get("Surname", ""), rng, config)

    if rng.random() < config["swap_name_prob"]:
        mutated["GivenName"], mutated["Surname"] = mutated["Surname"], mutated["GivenName"]

    birthday = row.get("Birthday", "")
    birthday_dt = parse_birthday(birthday)
    if birthday_dt:
        mutated["Birthday"] = format_birthday(birthday_dt, rng, config)
    elif rng.random() < config["missing_prob"]:
        mutated["Birthday"] = ""

    #mutated["uid"] = corrupt_uid(str(row.get("uid", "")), rng, config)
    return mutated


def iter_fakename_files(source_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(source_dir.glob("fakename_*.tsv")):
        name = path.name
        if any(tag in name for tag in ("_bf_encoded", "_tmh_encoded", "_tsh_encoded")):
            continue
        if name.endswith("_analysis.txt"):
            continue
        yield path


def process_file(path: pathlib.Path, output_dir: pathlib.Path, rng: random.Random, config: Dict[str, float]) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.with_suffix("").name
    output_path = output_path.with_name(output_path.name).with_suffix(".tsv")

    with path.open(newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        rows: List[Dict[str, str]] = [mutate_row(row, rng, config) for row in reader]
        fieldnames = reader.fieldnames or (list(rows[0].keys()) if rows else [])

    with output_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add controlled noise to FakeName TSV datasets.")
    parser.add_argument("--source-dir", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent, help="Directory containing fakename_*.tsv files.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / "noisy", help="Where to write noisy copies.")
    parser.add_argument("--noise-level", type=float, default=1.0, help="Scales how aggressive the corruption is (0 = none, 1 = default, >1 = heavier).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = build_noise_config(args.noise_level)

    source_dir = args.source_dir
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    files = list(iter_fakename_files(source_dir))
    if not files:
        raise SystemExit(f"No fakename TSV files found under {source_dir}")

    print(f"Writing noisy datasets to {args.output_dir} (noise level={args.noise_level})")
    for path in files:
        out_path = process_file(path, args.output_dir, rng, config)
        if out_path.is_absolute():
            try:
                pretty = out_path.relative_to(pathlib.Path.cwd())
            except ValueError:
                pretty = out_path
        else:
            pretty = out_path
        print(f"- {path.name} -> {pretty}")


if __name__ == "__main__":
    main()
