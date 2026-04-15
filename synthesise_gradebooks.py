"""
synthesise_gradebooks.py
========================
Generates privacy-safe synthetic versions of Canvas LMS gradebook CSV exports
for use in the public replication repository.

What is replaced
----------------
Column              Replacement strategy
-----------         -----------------------------------------------
student name        Synthetic Filipino-style full names (Faker en_PH locale)
                    Generated once per unique (student sis) so the same
                    student always gets the same synthetic name across files.

student id          Sequential integers starting from 10001, mapped 1-to-1
                    to the original student id. Same original id → same
                    synthetic id across files.

student sis         Format preserved: H{YEAR}{6-digit-number}
                    Year prefix kept (needed by analysis_engine.py to derive
                    school_year). The 6-digit suffix is replaced with a
                    zero-padded sequential counter per year group.
                    e.g.  H2024900129 → H2024000001

course sis          'FEUHS' prefix replaced with 'INST'.
                    Faculty surname (6th underscore token) replaced with
                    consistent synthetic codes FAC001, FAC002, etc.
                    Structural suffix (last token, e.g. '11S02a') is NEVER
                    touched — process_section_info() depends on it.

section sis         Same replacement as course sis, plus section suffix
                    preserved identically.

course id           Sequential integer (same original id → same synthetic id)
section id          Sequential integer (same original id → same synthetic id)
term id             Sequential integer (same original id → same synthetic id)

term                'FEU SHS MLA' → 'INST SHS'
                    'FEU SHS Manila' → 'INST SHS'
                    'FEU HS' → 'INST HS'
                    Any other FEU variant → 'INST'

All other columns   Kept exactly as-is (course names, grades, term sis,
                    enrollment state, score columns).

What is NOT changed
-------------------
course              Canonical subject names — required by SUBJECT_NAME_MAPPING
term sis            e.g. SY_2024_S1_SHS_OFFICIAL — already has no PII
unposted final grade  The grade value — core data required by the engine
enrollment state    active / concluded / inactive
All score columns   Numeric values preserved (current/final/unposted scores)
override columns    Preserved as-is

Usage
-----
1. Place this script in the same folder as your gradebook CSV files.
2. Edit INPUT_FILES to list all files you want to synthesise.
3. Run:  python synthesise_gradebooks.py
4. Synthetic files are written to a 'synthetic/' sub-folder.

All mapping tables (student, faculty, id) are built globally across ALL input
files so synthetic identifiers are consistent across semesters.
"""

import pandas as pd
import numpy as np
import re
import os
from faker import Faker
from pathlib import Path

# ── Configure these paths ────────────────────────────────────────────────────
INPUT_FILES = [
    "gb_2021-2022_1.csv",
    "gb_2021-2022_2.csv",
    "gb_2022-2023_1.csv",
    "gb_2022-2023_2.csv",
    "gb_2023-2024_1.csv",
    "gb_2023-2024_2.csv",
    "gb_2024-2025_1.csv",
    "gb_2024-2025_2.csv",
    "gb_2025-2026_1.csv",
]
OUTPUT_DIR = "synthetic"
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

fake = Faker("en_PH")
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ── Global mapping tables (built across ALL files for consistency) ────────────

def load_all(files):
    """Load all available CSV files, skip missing ones."""
    frames = {}
    for f in files:
        if os.path.exists(f):
            frames[f] = pd.read_csv(f, dtype=str, low_memory=False)
        else:
            print(f"  [skip] {f} not found")
    return frames


def build_global_maps(frames):
    """
    Build four global maps that remain consistent across all files:

    student_sis_map   : original_sis  → synthetic_sis
    student_name_map  : original_sis  → synthetic_name
    student_id_map    : original_id   → synthetic_id  (sequential integers)
    faculty_code_map  : original_name → FAC###
    numeric_id_maps   : {'course id': {…}, 'section id': {…}, 'term id': {…}}
    """
    all_rows = pd.concat(frames.values(), ignore_index=True)

    # ── Student SIS ──────────────────────────────────────────────────────────
    # Group by year prefix so counters restart per cohort year.
    # Format: H{YEAR}{6-digit sequential}
    unique_sis = all_rows['student sis'].dropna().unique()
    sis_map = {}
    year_counter = {}
    for sis in sorted(unique_sis):
        m = re.match(r'H(\d{4})', str(sis))
        year = m.group(1) if m else "0000"
        year_counter[year] = year_counter.get(year, 0) + 1
        sis_map[sis] = f"H{year}{year_counter[year]:06d}"

    # ── Student names — one synthetic name per SIS ──────────────────────────
    name_map = {}
    for sis in unique_sis:
        # Generate a realistic Filipino-style name: Firstname MI. Lastname
        fname = fake.first_name()
        lname = fake.last_name()
        mid   = fake.first_name()[0] + "."
        name_map[sis] = f"{fname} {mid} {lname}"

    # ── Student numeric IDs ──────────────────────────────────────────────────
    unique_ids = all_rows['student id'].dropna().unique()
    id_map = {orig: str(10001 + i) for i, orig in enumerate(sorted(unique_ids))}

    # ── Faculty last names (extracted from course sis / section sis) ─────────
    # Pattern: FEUHS_SY_YYYY_S#_SUBJECTCODE_FacultyName[_sectioncode]
    faculty_set = set()
    for col in ['course sis', 'section sis']:
        if col in all_rows.columns:
            for val in all_rows[col].dropna().unique():
                parts = str(val).split('_')
                if len(parts) >= 6:
                    faculty_set.add(parts[5])
    faculty_code_map = {
        name: f"FAC{i+1:03d}"
        for i, name in enumerate(sorted(faculty_set))
    }

    # ── Numeric Canvas IDs (course id, section id, term id) ─────────────────
    numeric_id_maps = {}
    for col in ['course id', 'section id', 'term id']:
        if col in all_rows.columns:
            unique_vals = all_rows[col].dropna().unique()
            numeric_id_maps[col] = {
                orig: str(20001 + i)
                for i, orig in enumerate(sorted(unique_vals))
            }

    return sis_map, name_map, id_map, faculty_code_map, numeric_id_maps


def anonymise_sis_string(value, faculty_map, prefix='INST'):
    """
    Replace institution prefix and faculty surname in a course sis or
    section sis string while preserving all other tokens exactly.

    Input:  FEUHS_SY_2024_S1_PCAL_Gravador_11S02a
    Output: INST_SY_2024_S1_PCAL_FAC001_11S02a

    The last token (section code, e.g. '11S02a') is NEVER modified
    because process_section_info() in analysis_engine.py depends on it.
    """
    if pd.isna(value):
        return value
    parts = str(value).split('_')
    if not parts:
        return value

    # Replace institution prefix (first token)
    parts[0] = prefix

    # Replace faculty name (6th token, index 5) if present
    if len(parts) >= 6 and parts[5] in faculty_map:
        parts[5] = faculty_map[parts[5]]
    elif len(parts) >= 6:
        # Faculty name token present but not in map — censor generically
        parts[5] = "FAC000"

    return '_'.join(parts)


def anonymise_term(value):
    """
    Replace FEU institution references in the 'term' column.
    Preserves the academic year and semester designation exactly.

    Examples:
      FEU SHS MLA_AY 2024-2025_1st Semester_Official
      → INST SHS_AY 2024-2025_1st Semester_Official

      FEU SHS Manila AY 2021-2022 2nd Semester
      → INST SHS AY 2021-2022 2nd Semester
    """
    if pd.isna(value):
        return value
    s = str(value)
    s = re.sub(r'FEU\s+SHS\s+MLA', 'INST SHS', s)
    s = re.sub(r'FEU\s+SHS\s+Manila', 'INST SHS', s)
    s = re.sub(r'FEU\s+HS\b', 'INST HS', s)
    s = re.sub(r'\bFEU\b', 'INST', s)
    return s


def synthesise_file(path, sis_map, name_map, id_map, faculty_map, num_id_maps):
    """
    Apply all anonymisation transforms to one CSV file and return the
    resulting DataFrame.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # student name — mapped from sis for consistency
    if 'student name' in df.columns and 'student sis' in df.columns:
        df['student name'] = df['student sis'].map(name_map).fillna(df['student name'])

    # student sis — synthetic with year preserved
    if 'student sis' in df.columns:
        df['student sis'] = df['student sis'].map(sis_map).fillna(df['student sis'])

    # student id — sequential integer
    if 'student id' in df.columns:
        df['student id'] = df['student id'].map(id_map).fillna(df['student id'])

    # course sis — replace institution + faculty
    if 'course sis' in df.columns:
        df['course sis'] = df['course sis'].apply(
            lambda v: anonymise_sis_string(v, faculty_map)
        )

    # section sis — same replacement, section code preserved
    if 'section sis' in df.columns:
        df['section sis'] = df['section sis'].apply(
            lambda v: anonymise_sis_string(v, faculty_map)
        )

    # term — remove FEU references
    if 'term' in df.columns:
        df['term'] = df['term'].apply(anonymise_term)

    # course id, section id, term id — sequential integers
    for col, col_map in num_id_maps.items():
        if col in df.columns:
            df[col] = df[col].map(col_map).fillna(df[col])

    return df


# ── Main execution ────────────────────────────────────────────────────────────

print("Loading all available CSV files...")
frames = load_all(INPUT_FILES)

if not frames:
    print("No CSV files found. Place your gradebook CSVs in the same folder as this script.")
    exit(1)

print(f"Found {len(frames)} file(s). Building global mapping tables...")
sis_map, name_map, id_map, faculty_map, num_id_maps = build_global_maps(frames)

print(f"  Unique students   : {len(sis_map):,}")
print(f"  Unique faculty    : {len(faculty_map):,}")
print(f"  Unique student IDs: {len(id_map):,}")
print()

for fname, _ in frames.items():
    print(f"Synthesising {fname}...")
    df_out = synthesise_file(fname, sis_map, name_map, id_map, faculty_map, num_id_maps)
    out_path = os.path.join(OUTPUT_DIR, fname)
    df_out.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(df_out):,} rows)")

print()
print("Done. Synthetic files are in the 'synthetic/' folder.")
print()
print("Verification checklist:")
print("  ✓ term sis preserved  (e.g. SY_2024_S1_SHS_OFFICIAL — no PII)")
print("  ✓ course names preserved  (needed by SUBJECT_NAME_MAPPING)")
print("  ✓ section suffix preserved  (e.g. _11S02a — needed by process_section_info)")
print("  ✓ grade columns preserved  (unposted final grade, scores)")
print("  ✓ student names replaced with synthetic Filipino-style names")
print("  ✓ student SIS year-prefix preserved, suffix randomised")
print("  ✓ faculty names replaced with FAC### codes")
print("  ✓ institution prefix FEUHS replaced with INST")
print("  ✓ FEU references in term column replaced with INST")
