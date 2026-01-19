import requests
import numpy as np
import pandas as pd
import math
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
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
        print("Ollama error:", response)
        return []

    return response["embeddings"]



def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            # "model": "deepseek-r1",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    response = r.json()
    print(response)
    return response

# ---------------- MAIN ----------------

# load saved dataframe
df = joblib.load("embeddings.joblib")

# ask question
incoming_query = input("Ask a Question: ")

# create embedding for question
question_embedding = create_embedding([incoming_query])[0]

# convert stored embeddings to numpy array
embedding_matrix = np.vstack(df["embedding"].values)

# find cosine similarity
similarities = cosine_similarity(
    embedding_matrix,
    [question_embedding]
).flatten()

#print(similarities)

# get top results
top_results = 3
Max_indx = similarities.argsort()[::-1][0:top_results]

#print(Max_indx)

# fetch matched rows
new_df = df.loc[Max_indx]

prompt = f''' I am Teaching web development in my Sigma web development course. Here are video
subtitle chunks containing video title, video number, start time in second, end time in second,
the text at that time:

{new_df[["title", "number", "start" ,"end",  "text"]].to_json(orient="records")}
-------------------------------------
"{incoming_query}"
Users asked  this question related to video chunks, you have to answer in human way (dont mention the above format, its just for you) where and how much
content is taught in which video (in which video and what timestamp) and guide the user to go
to the particular video. If user asked unrelated question, tell him you can only answer 
questions relted to course
'''

#print(new_df[["title", "number", "text"]])
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open ("response.txt", "w") as f:
    f.write(response)
# optional: print all rows (same as your original code)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
