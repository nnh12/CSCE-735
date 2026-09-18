#!/usr/bin/env python3
"""
Extract information about parallel programming with threads using OpenMP directives.
Reads PDF lecture slides and source code files from CSCE735 coursework.
"""

import re
import os
import csv
import textwrap
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent

# ── PDF extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages


def extract_omp_directives_from_text(pages):
    """Find OpenMP directives and related constructs in extracted text."""
    omp_pattern = re.compile(
        r'(#pragma\s+omp\s+\w+[^#\n]*)'
        r'|(\b omp_\w+\s*\([^)]*\))'
        r'|(\bparallel\s+for\b)'
        r'|(\bsections?\b.*\bsection\b)'
        r'|(\bsingle\b)'
        r'|(\bmaster\b)'
        r'|(\bcritical\b)'
        r'|(\bbarrier\b)'
        r'|(\batomic\b)'
        r'|(\breduction\b)'
        r'|(\bschedule\b\s*\([^)]*\))'
        r'|(\bnum_threads\b\s*\([^)]*\))'
        r'|(\bprivate\b\s*\([^)]*\))'
        r'|(\bshared\b\s*\([^)]*\))'
        r'|(\bfirstprivate\b\s*\([^)]*\))'
        r'|(\blastprivate\b\s*\([^)]*\))'
        r'|(\bdefault\s*\([^)]*\))'
        r'|(\bnowait\b)'
        r'|(\bordered\b)'
        r'|(\bcopyprivate\b\s*\([^)]*\))'
        r'|(\bcopyin\b\s*\([^)]*\))'
        r'|(\bflush\b)'
        r'|(\bthreadprivate\b)'
    )
    results = []
    for page_num, text in pages:
        for m in omp_pattern.finditer(text):
            matched = m.group(0).strip()
            if matched:
                results.append((page_num, matched))
    return results


def extract_threading_concepts(pages):
    """Extract key threading/OpenMP concepts mentioned in the slides."""
    concept_keywords = [
        "fork", "join", "parallel region", "work-sharing",
        "synchronization", "race condition", "deadlock", "data race",
        "mutual exclusion", "critical section", "barrier",
        "reduction", "schedule", "static", "dynamic", "guided",
        "Amdahl's law", "speedup", "efficiency", "scaling",
        "granularity", "load balancing", "overhead",
        "thread affinity", "NUMA", "false sharing",
        "task", "taskloop", "taskgroup", "taskwait",
        "nested parallelism", "OMP_NUM_THREADS",
        "omp_get_thread_num", "omp_get_num_threads",
        "omp_get_wtime", "omp_set_num_threads",
        "CancellationToken", "impedance matching",
    ]
    found = {}
    for page_num, text in pages:
        lower = text.lower()
        for kw in concept_keywords:
            if kw.lower() in lower:
                if kw not in found:
                    found[kw] = []
                found[kw].append(page_num)
    return found


# ── Source code extraction ──────────────────────────────────────────────────

def extract_omp_from_source(filepath):
    """Extract OpenMP directives and runtime calls from a source file."""
    directives = []
    runtime_calls = []
    pragma_omp = re.compile(r'^\s*#\s*pragma\s+omp\b(.*)', re.MULTILINE)
    omp_runtime = re.compile(r'\b(omp_\w+)\s*\(([^)]*)\)')

    with open(filepath) as f:
        content = f.read()

    for m in pragma_omp.finditer(content):
        directives.append(m.group(0).strip())

    for m in omp_runtime.finditer(content):
        runtime_calls.append(f"{m.group(1)}({m.group(2)})")

    return directives, runtime_calls


def extract_pthread_from_source(filepath):
    """Extract pthread usage from a source file."""
    calls = []
    pattern = re.compile(
        r'\b(pthread_\w+)\s*\(([^)]*)\)', re.MULTILINE
    )
    with open(filepath) as f:
        content = f.read()
    for m in pattern.finditer(content):
        calls.append(f"{m.group(1)}({m.group(2).strip()})")
    return calls


def extract_mpi_from_source(filepath):
    """Extract MPI usage from a source file."""
    calls = []
    pattern = re.compile(r'\b(MPI_\w+)\s*\(([^)]*)\)')
    with open(filepath) as f:
        content = f.read()
    for m in pattern.finditer(content):
        calls.append(f"{m.group(1)}({m.group(2).strip()})")
    return calls


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  CSCE735 — Parallel Programming with Threads & OpenMP")
    print("  Information Extraction Report")
    print("=" * 72)

    # ── 1. PDF slides ───────────────────────────────────────────────────
    pdf_files = sorted(ROOT.glob("x35-*.pdf"))
    print(f"\n[PDF SLIDES FOUND] {len(pdf_files)}")
    for p in pdf_files:
        print(f"  • {p.name}")

    all_omp_directives = []
    all_concepts = {}

    for pdf_file in pdf_files:
        pages = extract_text_from_pdf(pdf_file)
        print(f"\n{'─' * 72}")
        print(f"  {pdf_file.name}  ({len(pages)} pages with text)")
        print(f"{'─' * 72}")

        # Print first few pages as preview
        for pg, text in pages[:3]:
            wrapped = textwrap.indent(text, "  ").strip()
            print(f"\n  [Page {pg} preview]")
            for line in wrapped.splitlines()[:15]:
                print(f"    {line}")
            if len(text.splitlines()) > 15:
                print("    ...")

        # Extract OpenMP directives from slide text
        omp_hits = extract_omp_directives_from_text(pages)
        all_omp_directives.extend([(pdf_file.name, pg, d) for pg, d in omp_hits])
        if omp_hits:
            print(f"\n  OpenMP directives/calls found in slides:")
            seen = set()
            for pg, directive in omp_hits:
                key = directive[:60]
                if key not in seen:
                    print(f"    p.{pg:>3}: {directive}")
                    seen.add(key)

        # Extract threading concepts
        concepts = extract_threading_concepts(pages)
        for kw, pgs in concepts.items():
            if kw not in all_concepts:
                all_concepts[kw] = (pdf_file.name, pgs)
        if concepts:
            print(f"\n  Key concepts mentioned:")
            for kw, pgs in sorted(concepts.items()):
                pg_list = ",".join(str(p) for p in pgs[:5])
                if len(pgs) > 5:
                    pg_list += f"... (+{len(pgs)-5} more)"
                print(f"    {kw:<30s}  pages: {pg_list}")

    # ── 2. Source code files ────────────────────────────────────────────
    source_files = []
    for ext in ("*.c", "*.cpp", "*.h", "*.hpp"):
        source_files.extend(ROOT.rglob(ext))
    # exclude __MACOSX
    source_files = [f for f in source_files if "__MACOSX" not in str(f)]

    print(f"\n{'=' * 72}")
    print(f"  SOURCE CODE ANALYSIS  ({len(source_files)} files)")
    print(f"{'=' * 72}")

    summary_rows = []

    for src in sorted(source_files):
        rel = src.relative_to(ROOT)
        print(f"\n  ┌─ {rel}")

        # OpenMP
        omp_dir, omp_rt = extract_omp_from_source(src)
        if omp_dir:
            print(f"  │  OpenMP #pragma omp directives:")
            for d in omp_dir:
                print(f"  │    {d}")
        if omp_rt:
            print(f"  │  OpenMP runtime calls:")
            for r in omp_rt:
                print(f"  │    {r}")

        # Pthreads
        pthread_calls = extract_pthread_from_source(src)
        if pthread_calls:
            unique = sorted(set(pthread_calls))
            print(f"  │  pthread calls ({len(unique)} unique):")
            for c in unique[:15]:
                print(f"  │    {c}")
            if len(unique) > 15:
                print(f"  │    ... and {len(unique)-15} more")

        # MPI
        mpi_calls = extract_mpi_from_source(src)
        if mpi_calls:
            unique = sorted(set(mpi_calls))
            print(f"  │  MPI calls ({len(unique)} unique):")
            for c in unique:
                print(f"  │    {c}")

        if not omp_dir and not omp_rt and not pthread_calls and not mpi_calls:
            print(f"  │  (no parallel directives/calls found)")

        print(f"  └─")

        summary_rows.append({
            "file": str(rel),
            "omp_directives": len(omp_dir),
            "omp_runtime": len(omp_rt),
            "pthread_calls": len(pthread_calls),
            "mpi_calls": len(mpi_calls),
        })

    # ── 3. Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total source files scanned:  {len(source_files)}")
    total_omp_dir = sum(r["omp_directives"] for r in summary_rows)
    total_omp_rt = sum(r["omp_runtime"] for r in summary_rows)
    total_pthread = sum(r["pthread_calls"] for r in summary_rows)
    total_mpi = sum(r["mpi_calls"] for r in summary_rows)
    print(f"  Total OpenMP directives:     {total_omp_dir}")
    print(f"  Total OpenMP runtime calls:  {total_omp_rt}")
    print(f"  Total pthread calls:         {total_pthread}")
    print(f"  Total MPI calls:             {total_mpi}")

    print(f"\n  Threading models found across codebase:")
    if total_omp_dir > 0:
        print(f"    ✓ OpenMP (compiler directives for shared-memory parallelism)")
    if total_omp_rt > 0:
        print(f"    ✓ OpenMP runtime API (timing / thread queries)")
    if total_pthread > 0:
        print(f"    ✓ POSIX threads (pthreads — manual thread management)")
    if total_mpi > 0:
        print(f"    ✓ MPI (message passing for distributed-memory)")

    # ── 4. OpenMP concepts quick reference ──────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  OPENMP QUICK REFERENCE (from lecture slides)")
    print(f"{'=' * 72}")
    print(textwrap.dedent("""\
    OpenMP uses compiler directives (#pragma omp) to create parallel regions
    with fork/join parallelism. Key constructs:

    PARALLEL REGION:
      #pragma omp parallel [clause[,...]]
        { ... }  — team of threads created, work shared

    WORK-SHARING:
      #pragma omp for [schedule(static|dynamic|guided[,chunk])]
        — distributes loop iterations across threads
      #pragma omp sections / #pragma omp section
        — assigns different blocks to different threads
      #pragma omp single
        — one thread executes the block
      #pragma omp master
        — only the master (id=0) executes

    SYNCHRONIZATION:
      #pragma omp barrier        — all threads wait
      #pragma omp critical       — mutual exclusion for a block
      #pragma omp atomic         — atomic hardware update
      #pragma omp ordered        — sequential ordering of iterations

    DATA CLAUSES:
      private(var)       — per-thread copy, uninitialized
      shared(var)        — all threads share same copy
      firstprivate(var)  — per-thread copy, initialized from parent
      lastprivate(var)   — last iteration's value written back
      reduction(+:var)   — per-thread partial, combined at end

    TIMING:
      double omp_get_wtime()  — wall-clock time (seconds)

    THREAD MANAGEMENT:
      omp_get_thread_num()    — 0-based thread ID within team
      omp_get_num_threads()   — size of current team
      omp_set_num_threads(n)  — set team size for next parallel region

    SCHEDULING:
      static   — equal chunks, no overhead (default)
      dynamic  — threads grab chunks, good for uneven work
      guided   — decreasing chunk sizes, good for load imbalance
      runtime  — set via OMP_SCHEDULE environment variable
    """))

    # ── 5. Write CSV summary ────────────────────────────────────────────
    csv_path = ROOT / "omp_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "omp_directives", "omp_runtime",
            "pthread_calls", "mpi_calls"
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  CSV summary written to: {csv_path.name}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
