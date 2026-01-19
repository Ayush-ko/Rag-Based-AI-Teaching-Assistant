import requests
import os
import json
import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # Clean text list (VERY IMPORTANT)
    clean_texts = []
    for t in text_list:
        if t is None:
            continue
        if isinstance(t, float) and math.isnan(t):
            continue
        t = str(t).replace("\x00", "").strip()
        if t:
            clean_texts.append(t[:6000])

    if not clean_texts:
        return []

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": clean_texts
        }
    )

    response = r.json()

    if "embeddings" not in response:
        print("❌ Ollama error:", response)
        return []

    return response["embeddings"]


# ---------------- MAIN ----------------
jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    texts = []
    valid_chunks = []

    for chunk in content["chunks"]:
        texts.append(chunk.get("text"))
        valid_chunks.append(chunk)

    embeddings = create_embedding(texts)

    if len(embeddings) != len(valid_chunks):
        print(f"⚠ Skipping {json_file} (embedding mismatch)")
        continue

    for i, chunk in enumerate(valid_chunks):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
   
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)

df = pd.DataFrame.from_records(my_dicts)
# save this dataframe
joblib.dump(df, "embeddings.joblib")

# print(df)


# print(question_embedding)

# print(np.vstack(df["embedding"].values))
# print(np.vstack(df["embedding"]).shape)
#find similarities of question_embedding with other embeddings


