import os

# Delete old cache files
cache_files = [
    r"C:\Users\Shivam Kumar\frenemy\TruthCheck\data\verified_facts.faiss",
    r"C:\Users\Shivam Kumar\frenemy\TruthCheck\data\verified_facts_cache.json",
]

for file in cache_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted: {file}")
    else:
        print(f"Not found: {file}")

print("\nNow run your pipeline again!")