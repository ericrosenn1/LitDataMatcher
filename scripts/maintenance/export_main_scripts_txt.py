"""Bundle canonical package scripts into one readable text file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT = Path.home() / "Downloads" / "LitDataMatcher_main_scripts.txt"


def repo_root_from_script() -> Path:
    """Return the repository root when this script is run in-place."""

    return Path(__file__).resolve().parents[2]


def iter_main_scripts(repo_root: Path) -> list[Path]:
    """Return canonical production Python files in deterministic order."""

    package_dir = repo_root / "litdatamatcher"
    return sorted(path for path in package_dir.glob("*.py") if path.is_file())


def format_file_block(path: Path, repo_root: Path) -> str:
    """Render one source file with an inspectable relative filename header."""

    relative = path.relative_to(repo_root).as_posix()
    content = path.read_text(encoding="utf-8")
    return (
        "\n"
        + "=" * 88
        + f"\nFILE: {relative}\n"
        + "=" * 88
        + "\n\n"
        + content.rstrip()
        + "\n"
    )


def export_main_scripts(repo_root: Path, output_path: Path) -> Path:
    """Write all canonical package scripts into one text document."""

    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    scripts = iter_main_scripts(repo_root)
    if not scripts:
        raise FileNotFoundError(f"No Python files found in {repo_root / 'litdatamatcher'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "LitDataMatcher main scripts bundle",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source root: {repo_root}",
        f"File count: {len(scripts)}",
        "",
        "Included files:",
        *[f"- {path.relative_to(repo_root).as_posix()}" for path in scripts],
        "",
    ]
    body = "".join(format_file_block(path, repo_root) for path in scripts)
    output_path.write_text("\n".join(header) + body, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options for the exporter."""

    parser = argparse.ArgumentParser(
        description="Export the canonical LitDataMatcher Python scripts into one text file."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(),
        help="Repository root containing the litdatamatcher package.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination text file.",
    )
    return parser


def main() -> int:
    """Run the main-script text exporter."""

    args = build_parser().parse_args()
    output_path = export_main_scripts(args.repo_root, args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
