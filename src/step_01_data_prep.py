import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

from step_00_functions import *

# load the data
data_file_path = (Path(__file__).parent / "../data/raw/BioASQ-training4b/BioASQ-trainingDataset4b.json").resolve()

with open(data_file_path) as f:
    data = json.load(f)

# randomly select ~10% of the questions
num_questions = len(data['questions'])
subset_size = max(1, num_questions // 10)
subset_questions = random.sample(data['questions'], subset_size)


structured_data = []

for question in subset_questions:
    q_text = question['body']
    relevant_pmids = set(question['documents'])

    for pmid in relevant_pmids:
        abstract = fetch_abstract(pmid)
        if abstract:
            combined_text = f"Question: {q_text}\nAbstract: {abstract}"
            embedding = get_embedding(combined_text)
            structured_data.append({
                "question": q_text,
                "abstract": abstract,
                "embedding": embedding,
                "label": 1
            })

df = pd.DataFrame(structured_data)

# Save data
project_root = Path(__file__).parent.parent.resolve()
export_file_path = project_root / "data" / "processed" / "bioasq_embeddings_subset.parquet"
df.to_parquet(export_file_path, index=False)

# test_df = pd.DataFrame({'test_column': ["Don't care"]})
# test_file_path = project_root / "data" / "processed" / "test_df.csv"
# test_df.to_csv(test_file_path)

if __name__ == '__main__':
    # print(f"Using subset of {len(subset_questions)} questions out of {num_questions}")
    # print(fetch_abstract("12345678"))
    print(f"File saved successfully? {export_file_path.exists()}")