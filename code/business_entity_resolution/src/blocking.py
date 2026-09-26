import duckdb
import os
import time

def run_blocking(
    con: duckdb.DuckDBPyConnection,
    s1_table: str,
    target_table: str,
    candidates_table: str = "candidates_all"
):
    """
    Execute high-speed, multi-pass equi-join candidate generation in DuckDB.
    Strictly partitioned by country with 13 complementary high-recall passes:
      - 2-token prefix match (p1 + p2)
      - 1st token + Postal code
      - 1st token + Door number
      - Compact name 5-char prefix (handles domain names & merged tokens)
      - Single long core name word (p1 >= 5)
      - Address Anchor: Door + Postal code (same building)
      - Address Anchor: Door + Street token 1 (catches Indic script & DBA name changes)
      - Address Anchor: Door + Street token 2
      - Word permutation: S1 p2 = Target p1 (e.g. 'Hotel Sheron' vs 'Sheron')
      - Word permutation: S1 p1 = Target p2
      - Word permutation: S1 p3 = Target p1
      - Postal match + 4-char compact name prefix
      - Door match + 4-char compact name prefix
    """
    print(f"Running high-recall multi-pass blocking: {s1_table} x {target_table} -> {candidates_table}...", flush=True)
    t0 = time.time()
    
    con.execute(f"""
    CREATE OR REPLACE TABLE {candidates_table} AS
    -- Pass 1: Country + 2-token prefix match (p1 + p2)
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p1 = t.p1 AND s.p2 = t.p2
    WHERE length(s.p1) >= 3 AND length(s.p2) >= 3

    UNION

    -- Pass 2: Country + 1st token + Postal code
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p1 = t.p1 AND s.postal = t.postal
    WHERE length(s.p1) >= 3 AND s.postal != ''

    UNION

    -- Pass 3: Country + 1st token + Door number
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p1 = t.p1 AND s.door = t.door
    WHERE length(s.p1) >= 3 AND s.door != ''

    UNION

    -- Pass 4: Compact name prefix 5-char (catches domain names, merged text, typos)
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.cmp5 = t.cmp5
    WHERE length(s.cmp5) >= 5

    UNION

    -- Pass 5: Single long core name word (p1 >= 5)
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p1 = t.p1
    WHERE length(s.p1) >= 5

    UNION

    -- Pass 6: Address Anchor: Door + Postal code (identical building)
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
    WHERE s.postal != '' AND length(s.door) >= 2

    UNION

    -- Pass 7: Address Anchor: Door + Street token 1 (catches Indic script & DBA name changes)
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.door = t.door AND s.a1 = t.a1
    WHERE length(s.door) >= 2 AND length(s.a1) >= 4

    UNION

    -- Pass 8: Address Anchor: Door + Street token 2
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.door = t.door AND s.a2 = t.a2
    WHERE length(s.door) >= 2 AND length(s.a2) >= 4

    UNION

    -- Pass 9: Word permutation: S1 p2 = Target p1 (e.g. 'Hotel Sheron' vs 'Sheron')
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p2 = t.p1
    WHERE length(s.p2) >= 4

    UNION

    -- Pass 10: Word permutation: S1 p1 = Target p2
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p1 = t.p2
    WHERE length(s.p1) >= 4

    UNION

    -- Pass 11: Word permutation: S1 p3 = Target p1
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.p3 = t.p1
    WHERE length(s.p3) >= 4

    UNION

    -- Pass 12: Postal match + 4-char compact name prefix
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.postal = t.postal AND s.cmp4 = t.cmp4
    WHERE s.postal != '' AND length(s.cmp4) >= 4

    UNION

    -- Pass 13: Door match + 4-char compact name prefix
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s
    JOIN {target_table} t ON s.country = t.country AND s.door = t.door AND s.cmp4 = t.cmp4
    WHERE length(s.door) >= 2 AND length(s.cmp4) >= 4;
    """)
    
    total_pairs = con.execute(f"SELECT count(*) FROM {candidates_table}").fetchone()[0]
    elapsed = time.time() - t0
    print(f"Blocking completed in {elapsed:.2f}s: generated {total_pairs:,} candidate pairs.", flush=True)
    return total_pairs
