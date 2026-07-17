#!/usr/bin/env python3
"""Reconstruct the local PDB dataset from the tracked manifest.

The repository tracks only ``data/pdb_ids.txt``. Structure files are downloaded
from RCSB PDB and written using the historical ``pdb<id>.ent`` naming convention.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "pdb_ids.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pdb_files"
PDB_ID_PATTERN = re.compile(r"^[0-9][a-z0-9]{3}$")
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
REQUEST_TIMEOUT_SECONDS = 30
CHUNK_SIZE = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDB structures listed in data/pdb_ids.txt."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to the PDB ID manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for downloaded .ent files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing non-empty .ent files instead of skipping them",
    )
    return parser.parse_args(argv)


def read_manifest(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as manifest:
        return [line.strip().lower() for line in manifest if line.strip()]


def validate_pdb_id(pdb_id: str) -> None:
    if not PDB_ID_PATTERN.fullmatch(pdb_id):
        raise ValueError(
            f"{pdb_id!r} is not a valid four-character PDB identifier"
        )


def destination_for(output_dir: Path, pdb_id: str) -> Path:
    return output_dir / f"pdb{pdb_id}.ent"


def download_structure(pdb_id: str, destination: Path) -> None:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    request = Request(
        url,
        headers={"User-Agent": "helixscope-pdb-downloader/1.0"},
    )
    temp_path: Path | None = None

    try:
        with (
            urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file,
        ):
            temp_path = Path(temp_file.name)
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)

        if temp_path is None or temp_path.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def process_manifest(
    pdb_ids: list[str],
    *,
    output_dir: Path,
    overwrite: bool,
) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    failed = 0
    seen: set[str] = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    for pdb_id in pdb_ids:
        if pdb_id in seen:
            continue
        seen.add(pdb_id)

        try:
            validate_pdb_id(pdb_id)
            destination = destination_for(output_dir, pdb_id)
            if (
                destination.exists()
                and destination.stat().st_size > 0
                and not overwrite
            ):
                skipped += 1
                continue

            download_structure(pdb_id, destination)
            downloaded += 1
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"failed {pdb_id}: {error}", file=sys.stderr)

    return downloaded, skipped, failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        pdb_ids = read_manifest(args.manifest)
    except OSError as error:
        print(f"could not read manifest {args.manifest}: {error}", file=sys.stderr)
        return 2

    downloaded, skipped, failed = process_manifest(
        pdb_ids,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        "summary: "
        f"downloaded={downloaded} "
        f"skipped={skipped} "
        f"failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
