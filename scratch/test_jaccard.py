import duckdb
con = duckdb.connect()
r = con.execute("""
SELECT 
    list_intersect(['apple', 'store'], ['apple', 'inc', 'store']) as inter,
    len(list_intersect(['apple', 'store'], ['apple', 'inc', 'store'])) as inter_len,
    len(list_distinct(list_concat(['apple', 'store'], ['apple', 'inc', 'store']))) as union_len,
    len(list_intersect(['apple', 'store'], ['apple', 'inc', 'store']))::float /
    nullif(len(list_distinct(list_concat(['apple', 'store'], ['apple', 'inc', 'store']))), 0) as jaccard
""").fetchdf()
print(r)
