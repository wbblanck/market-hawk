"""Load encrypted capture files from the ``market-data`` branch into a local DuckDB.

    git fetch origin market-data && git worktree add ../market-data FETCH_HEAD
    export CAPTURE_PASSPHRASE=...            # the same value as the repository secret
    python scripts/import_captures.py ../market-data/captures --db data/nfl_market.duckdb

Idempotent: snapshot ids and run ids already stored are skipped. Decrypts with the
``openssl`` command line (AES-256-CBC, PBKDF2, 200,000 iterations), the same settings the
workflow encrypts with. Plain ``.jsonl.gz`` files (local test runs) are loaded as they are.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_market.collectors.capture import import_export_file  # noqa: E402
from nfl_market.config import settings  # noqa: E402


def decrypt(src: Path, dest: Path) -> None:
    if not os.environ.get("CAPTURE_PASSPHRASE"):
        raise SystemExit("set CAPTURE_PASSPHRASE to decrypt .enc capture files")
    subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-in", str(src),
                    "-out", str(dest), "-pass", "env:CAPTURE_PASSPHRASE"], check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Import Market Hawk capture files into DuckDB")
    p.add_argument("path", help="a capture file or a directory searched recursively")
    p.add_argument("--db", default=settings.database_path)
    a = p.parse_args(argv)
    root = Path(a.path)
    files = [root] if root.is_file() else sorted([*root.rglob("*.jsonl.gz.enc"), *root.rglob("*.jsonl.gz")])
    totals = {"collection_runs": 0, "market_snapshots": 0, "quote_snapshots": 0}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            plain = f
            if f.suffix == ".enc":
                plain = Path(tmp) / f.name.removesuffix(".enc")
                decrypt(f, plain)
            for k, v in import_export_file(plain, a.db).items():
                totals[k] += v
    print(f"{len(files)} files; newly inserted {totals} into {a.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
