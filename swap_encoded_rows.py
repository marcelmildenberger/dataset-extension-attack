"""
Utility script to introduce encoding-level misalignment in encoded FakeName datasets.

It simulates a bug where the encoded representations are occasionally attached
to the wrong record (while the UID stays the same in the encoded file).

Example:
    python swap_encoded_rows.py --input-dir data/datasets/noisy --output-dir data/datasets/noisy --swap-prob 0.01
"""

import argparse
import csv
import pathlib
import random
from typing import Dict, Iterable, List, Sequence


def iter_encoded_files(input_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """
    Yield encoded fakename files under input_dir.

    We treat files matching:
        fakename_*_bfd_encoded.tsv
        fakename_*_bf_encoded.tsv
        fakename_*_tmh_encoded.tsv
        fakename_*_tsh_encoded.tsv
    as encoded datasets.
    """
    patterns = [
        "fakename_*_bf_encoded.tsv",
        "fakename_*_bfd_encoded.tsv",
        "fakename_*_tmh_encoded.tsv",
        "fakename_*_tsh_encoded.tsv",
    ]
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            yield path


def get_uid_and_encoding_column(fieldnames: Sequence[str]) -> (str, str):
    """
    Identify the UID column and the single encoding column.

    Assumptions:
      - There is a column named 'uid', or UID is the last column.
      - The encoding column is always the column immediately before UID and
        has a name like 'bloomfilter', 'tabminhash', or 'twostephash'.
    """
    if not fieldnames:
        raise ValueError("No fieldnames found in encoded file.")

    # Determine UID column: either explicitly named 'uid' or the last column.
    if "uid" in fieldnames:
        uid_col = "uid"
        uid_index = fieldnames.index(uid_col)
    else:
        uid_index = len(fieldnames) - 1
        uid_col = fieldnames[uid_index]

    if uid_index == 0:
        raise ValueError("UID column cannot be the first/only column.")

    # The encoding column is always the column directly before UID.
    enc_col = fieldnames[uid_index - 1]
    return uid_col, enc_col


def apply_encoding_swaps(
    rows: List[Dict[str, str]],
    enc_col: str,
    rng: random.Random,
    swap_prob: float,
) -> None:
    """
    In-place: with probability `swap_prob` per random pair of rows,
    swap the encoding column between the two rows while keeping UID and all
    preceding columns (GivenName, Surname, Birthday, etc.) fixed.

    Conceptually:
        Row i: [GivenName_i, Surname_i, Birthday_i, enc_i, uid_i]
        Row j: [GivenName_j, Surname_j, Birthday_j, enc_j, uid_j]

    After swap:
        Row i: [GivenName_i, Surname_i, Birthday_i, enc_j, uid_i]
        Row j: [GivenName_j, Surname_j, Birthday_j, enc_i, uid_j]
    """
    if not rows:
        return

    if enc_col not in rows[0]:
        # Nothing to do if encoding column does not exist.
        return

    if swap_prob <= 0.0:
        return

    indices = list(range(len(rows)))
    rng.shuffle(indices)

    # Walk through indices in pairs (i, i+1)
    for k in range(0, len(indices) - 1, 2):
        if rng.random() >= swap_prob:
            continue

        i, j = indices[k], indices[k + 1]
        row_i, row_j = rows[i], rows[j]

        # Swap only the encoding column; do NOT touch uid or other data.
        row_i[enc_col], row_j[enc_col] = row_j[enc_col], row_i[enc_col]


def process_encoded_file(
    path: pathlib.Path,
    output_dir: pathlib.Path,
    rng: random.Random,
    swap_prob: float,
) -> pathlib.Path:
    """
    Load an encoded TSV, apply encoding swaps, and write to output_dir
    with a suffix added before the extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with path.open(newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = reader.fieldnames or (list(rows[0].keys()) if rows else [])

    if not rows:
        # just copy the header to the new file
        output_path = output_dir / path.with_suffix("").name
        output_path = output_path.with_name(output_path.name).with_suffix(".tsv")
        with output_path.open("w", newline="") as dst:
            if fieldnames:
                writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
        return output_path

    uid_col, enc_col = get_uid_and_encoding_column(fieldnames)

    # Apply in-place swaps on the single encoding column.
    apply_encoding_swaps(rows, enc_col, rng, swap_prob)

    # Write out the corrupted encoded dataset
    output_path = output_dir / path.with_suffix("").name
    output_path = output_path.with_name(output_path.name).with_suffix(".tsv")

    with output_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly swap encodings between records in encoded FakeName datasets "
            "to simulate post-encoding misalignment."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing encoded fakename_*.tsv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where encoding-swapped copies will be written.",
    )
    parser.add_argument(
        "--swap-prob",
        type=float,
        default=0.005,
        help="Probability per random pair of rows to swap encodings (default: 0.005 = 0.5%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = list(iter_encoded_files(args.input_dir))
    if not files:
        raise SystemExit(f"No encoded fakename TSV files found under {args.input_dir}")

    print(
        f"Applying encoding swaps to {len(files)} files "
        f"(swap_prob={args.swap_prob}) and writing to {args.output_dir}"
    )

    for path in files:
        out_path = process_encoded_file(
            path=path,
            output_dir=args.output_dir,
            rng=rng,
            swap_prob=args.swap_prob
        )
        try:
            pretty = out_path.relative_to(pathlib.Path.cwd())
        except ValueError:
            pretty = out_path
        print(f"- {path.name} -> {pretty}")


if __name__ == "__main__":
    main()