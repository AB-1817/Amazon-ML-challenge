import duckdb, time

gt_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv'
s1_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv'

con = duckdb.connect(':memory:')
con.execute("SET threads=8; SET memory_limit='6GB';")

t0 = time.time()
con.execute(f"""
CREATE TABLE s1_sample AS
SELECT entity_id, country, business_name, business_address
FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')
USING SAMPLE 150000 (reservoir, 42);
""")

con.execute(f"""
CREATE TABLE gt_sample AS
SELECT s.entity_id, g.matched_entity_ids
FROM s1_sample s
LEFT JOIN read_csv_auto('{gt_path}', delim='\t', header=true, quote='') g
ON s.entity_id = g.source1_entity_id;
""")

stats = con.execute("""
SELECT 
    count(*) as total_s1,
    count(matched_entity_ids) as matched_count,
    sum(case when matched_entity_ids is null or trim(matched_entity_ids) = '' then 1 else 0 end) as singleton_count
FROM gt_sample;
""").fetchone()

print(f"Total S1: {stats[0]}, With matches: {stats[1]}, Singletons: {stats[2]} in {time.time()-t0:.2f}s")
