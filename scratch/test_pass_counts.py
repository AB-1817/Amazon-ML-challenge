import duckdb, time

con = duckdb.connect(':memory:')
s1_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv'
s2_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv'

con.execute(f"""
CREATE TABLE s1_test AS
SELECT entity_id, country, business_name, business_address
FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')
WHERE country = 'India' LIMIT 1000;
""")

con.execute(f"""
CREATE TABLE s2_test AS
SELECT entity_id, country, business_name, business_address
FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='')
WHERE country = 'India' LIMIT 100000;
""")

print("Testing pass counts on 1,000 S1 x 100,000 S2...")
# Let's count pairs per pass
passes = [
    ("Pass 1 (p1+p2)", "length(split_part(s.business_name,' ',1))>=3 and length(split_part(s.business_name,' ',2))>=3 and lower(split_part(s.business_name,' ',1))=lower(split_part(t.business_name,' ',1)) and lower(split_part(s.business_name,' ',2))=lower(split_part(t.business_name,' ',2))"),
    ("Pass 4 (cmp5)", "length(replace(s.business_name,' ',''))>=5 and lower(left(replace(s.business_name,' ',''),5))=lower(left(replace(t.business_name,' ',''),5))"),
    ("Pass 5 (p1>=5)", "length(split_part(s.business_name,' ',1))>=5 and lower(split_part(s.business_name,' ',1))=lower(split_part(t.business_name,' ',1))")
]

for name, cond in passes:
    t0 = time.time()
    cnt = con.execute(f"SELECT count(*) FROM s1_test s JOIN s2_test t ON s.country=t.country AND {cond}").fetchone()[0]
    print(f"{name}: {cnt:,} pairs in {time.time()-t0:.2f}s")
