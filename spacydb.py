import os
from config import DATA_DIR

# Delete old cache files
cache_files = [
    str(DATA_DIR / "verified_facts.faiss"),
    str(DATA_DIR / "verified_facts_cache.json"),
]

for file in cache_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted: {file}")
    else:
        print(f"Not found: {file}")

print("\nNow run your pipeline again!")
