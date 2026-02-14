import pandas as pd
import json
import os

def convert_csv_to_jsonl(csv_path="data/lifestyle_patterns.csv", jsonl_path="data/lifestyle_patterns.jsonl"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Create it first!")
        return

    df = pd.read_csv(csv_path)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Convert "warm water" -> [{"LOWER": "warm"}, {"LOWER": "water"}]
            pattern_text = str(row['pattern']).strip()
            pattern_tokens = [{"LOWER": word.lower()} for word in pattern_text.split()]
            
            line = {"label": row['label'], "pattern": pattern_tokens}
            f.write(json.dumps(line) + '\n')
            
    print(f"--- Successfully created {jsonl_path} with {len(df)} patterns ---")

if __name__ == "__main__":
    convert_csv_to_jsonl()