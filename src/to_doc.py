import json

import pandas as pd


df = pd.read_csv("data/SkincareAddiction.csv")
print(df.columns)

# Create documents for LangChain
documents = []

for idx, row in df.iterrows():
    #
    # skip documents without text (nan)
    if isinstance(row["selftext"], str):

        text = f"""
        Title:
        {row['title']}

        Content:
        {row['selftext']}
        """

        documents.append({
            "id": row["id"],
            "text": text,
            "metadata": {
                "score": row["score"],
                "comments": row["num_comments"]
            }
        })


# Write each document to file
for i, doc in enumerate(documents):
    
    with open("data/processed/Skincare" + str(i), 'w') as f:
        json.dump(doc, f, indent=4)