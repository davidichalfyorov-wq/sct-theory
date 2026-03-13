#!/usr/bin/env python
"""
SCT Theory — Publication Build Script

Compiles LaTeX documents using tectonic. Can build individual files,
all theory documents, or paper drafts.

Usage:
    python papers/build.py                          # build all theory docs
    python papers/build.py theory/axioms/SCT_postulates.tex  # build one file
    python papers/build.py --drafts                 # build papers/drafts/*.tex
    python papers/build.py --check                  # dry-run: list what would compile
    python papers/build.py --clean                  # remove build artifacts

Requires tectonic in PATH (installed via scoop or .tools/msvc/).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
_IS_WINDOWS = os.name == "nt"
_WINDOWS_FONTCONFIG_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "GIMP 2" / "etc" / "fonts" / "fonts.conf",
    Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "GIMP 2" / "32" / "etc" / "fonts" / "fonts.conf",
)
PDF_CREDITS = "Aliaksandr Samatyia contributed research-assistance and workflow support."
PDF_METADATA_EXCLUDE_PREFIXES = (
    ("papers", "latex-sources"),
    ("papers", "references"),
)

# All known compilable .tex files
THEORY_DOCS = sorted(
    list((ROOT / "theory").rglob("*.tex"))
    + list((ROOT / "docs").rglob("*.tex"))
    + list((ROOT / "data" / "experimental").rglob("*.tex"))
)

DRAFT_DIR = ROOT / "papers" / "drafts"


def find_tectonic():
    """Find tectonic binary."""
    # Try system PATH first
    try:
        result = subprocess.run(
            ["tectonic", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=build_tectonic_env(),
        )
        if result.returncode == 0:
            return "tectonic"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try project-local binary
    local = ROOT / ".tools" / "msvc" / "tectonic.exe"
    if local.exists():
        return str(local)

    return None


def find_fontconfig_file(env=None):
    """Resolve a usable FONTCONFIG_FILE for Windows tectonic runs."""
    base_env = os.environ if env is None else env
    configured = base_env.get("FONTCONFIG_FILE")
    if configured and Path(configured).exists():
        return str(Path(configured))

    if not _IS_WINDOWS:
        return None

    for candidate in _WINDOWS_FONTCONFIG_CANDIDATES:
        if Path(candidate).exists():
            return str(Path(candidate))

    return None


def build_tectonic_env(env=None):
    """Build subprocess env for tectonic with a stable fontconfig path."""
    base_env = dict(os.environ if env is None else env)
    fontconfig = find_fontconfig_file(base_env)
    if fontconfig:
        base_env["FONTCONFIG_FILE"] = fontconfig
    return base_env


def compile_tex(tex_path, tectonic_bin):
    """Compile a single .tex file. Returns (success, elapsed, output)."""
    tex_path = Path(tex_path).resolve()
    if not tex_path.exists():
        return False, 0, f"File not found: {tex_path}"

    t0 = time.time()
    result = subprocess.run(
        [tectonic_bin, str(tex_path.name)],
        cwd=str(tex_path.parent),
        capture_output=True,
        text=True,
        timeout=120,
        env=build_tectonic_env(),
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    # On Windows, tectonic may fail with OS error 32 if PDF is open in a viewer
    if result.returncode != 0 and "os error 32" in output.lower():
        pdf = tex_path.with_suffix(".pdf")
        if pdf.exists():
            return False, elapsed, output + "\n(PDF locked by another process - close viewer and retry)"
    return result.returncode == 0, elapsed, output


def annotate_pdf_credits(pdf_path):
    """Embed repository credit metadata without changing visible page content."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    metadata = {}
    if reader.metadata:
        for key, value in reader.metadata.items():
            if not key:
                continue
            metadata[str(key)] = "" if value is None else str(value)
    metadata["/Credits"] = PDF_CREDITS
    writer.add_metadata(metadata)

    with NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(pdf_path.parent)) as handle:
        temp_path = Path(handle.name)
    with temp_path.open("wb") as output:
        writer.write(output)
    temp_path.replace(pdf_path)


def annotate_project_pdfs():
    """Apply credit metadata to all project-owned PDFs."""
    updated = 0
    for pdf_path in ROOT.rglob("*.pdf"):
        rel_parts = pdf_path.relative_to(ROOT).parts
        if any(rel_parts[: len(prefix)] == prefix for prefix in PDF_METADATA_EXCLUDE_PREFIXES):
            continue
        annotate_pdf_credits(pdf_path)
        updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="SCT Theory publication builder")
    parser.add_argument("files", nargs="*", help="Specific .tex files to build")
    parser.add_argument("--drafts", action="store_true", help="Build papers/drafts/")
    parser.add_argument("--check", action="store_true", help="Dry-run: list targets")
    parser.add_argument("--clean", action="store_true",
                        help="Remove .aux, .log, .out, .toc artifacts")
    args = parser.parse_args()

    tectonic = find_tectonic()
    if tectonic is None and not args.check and not args.clean:
        print("ERROR: tectonic not found. Install via: scoop install tectonic")
        sys.exit(1)

    # Determine targets
    if args.files:
        targets = [ROOT / f for f in args.files]
    elif args.drafts:
        targets = sorted(DRAFT_DIR.glob("*.tex")) if DRAFT_DIR.exists() else []
        if not targets:
            print("No .tex files in papers/drafts/")
            sys.exit(0)
    else:
        targets = THEORY_DOCS

    if args.clean:
        extensions = {".aux", ".log", ".out", ".toc", ".synctex.gz", ".fls",
                      ".fdb_latexmk"}
        count = 0
        for tex in targets:
            for ext in extensions:
                artifact = tex.with_suffix(ext)
                if artifact.exists():
                    artifact.unlink()
                    count += 1
        print(f"Cleaned {count} build artifacts")
        return

    if args.check:
        print(f"Would compile {len(targets)} file(s):")
        for t in targets:
            rel = t.relative_to(ROOT)
            pdf = t.with_suffix(".pdf")
            status = "exists" if pdf.exists() else "MISSING"
            print(f"  [{status}] {rel}")
        return

    # Build
    print(f"Building {len(targets)} file(s) with {tectonic}")
    results = []
    for tex in targets:
        rel = tex.relative_to(ROOT)
        ok, elapsed, output = compile_tex(tex, tectonic)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {rel} ({elapsed:.1f}s)")
        if not ok:
            # Print last few lines of error output
            for line in output.strip().split("\n")[-5:]:
                print(f"    {line}")
        results.append((rel, ok, elapsed))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total_time = sum(t for _, _, t in results)
    print(f"\n{passed}/{len(results)} compiled successfully ({total_time:.1f}s)")
    if failed:
        print(f"{failed} FAILED:")
        for name, ok, _ in results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)

    annotated = annotate_project_pdfs()
    print(f"Embedded PDF credit metadata in {annotated} file(s)")


if __name__ == "__main__":
    main()
