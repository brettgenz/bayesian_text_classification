import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

from step_00_functions import *

start_time = time.time()

# functions
def pick_irrelevant_question(main_question, all_questions, all_relevant_pmids):
    irrelevant_candidates = [
        q for q in all_questions if q['id'] != main_question['id'] and
        len(set(q['documents']).intersection(all_relevant_pmids)) == 0
    ]
    return random.choice(irrelevant_candidates) if irrelevant_candidates else None

# load the data
project_root = Path(__file__).parent.parent.resolve()
data_path = project_root / "data" / "raw" / "BioASQ-training4b" / "BioASQ-trainingDataset4b.json"

with open(data_path) as f:
    data = json.load(f)

# randomly select ~10% of the questions as relevant
random.seed(8675309)

num_questions = len(data['questions'])
subset_size = max(1, num_questions // 100)
subset_questions = random.sample(data['questions'], subset_size)
nonselected_questions = [q for q in data['questions'] if q not in subset_questions]

# create reference set of all relevant PMIDs
all_relevant_pmids = set(pmid for question in subset_questions for pmid in question['documents'])

# build data set of relevant questions
structured_data = []

for main_question in subset_questions:
    q_text = main_question['body']
    relevant_pmids = main_question['documents'][:10]  # Limit to 10 relevant abstracts

    # Relevant abstracts (label=1)
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

    # Pick one clearly irrelevant question
    irrelevant_question = pick_irrelevant_question(main_question, nonselected_questions, all_relevant_pmids)

    if irrelevant_question:
        irrelevant_pmids = irrelevant_question['documents'][:10]

        # Irrelevant abstracts (label=0)
        for pmid in irrelevant_pmids:
            abstract = fetch_abstract(pmid)
            if abstract:
                combined_text = f"Question: {q_text}\nAbstract: {abstract}"
                embedding = get_embedding(combined_text)
                structured_data.append({
                    "question": q_text,
                    "abstract": abstract,
                    "embedding": embedding,
                    "label": 0
                })

df = pd.DataFrame(structured_data)

# Save data
export_file_path = project_root / "data" / "processed" / "bioasq_embeddings_subset.parquet"
df.to_parquet(export_file_path, index=False)

num_records = len(df)

end_time = time.time()
elapsed = end_time - start_time

# test_df = pd.DataFrame({'test_column': ["Don't care"]})
# test_file_path = project_root / "data" / "processed" / "test_df.csv"
# test_df.to_csv(test_file_path)

if __name__ == '__main__':
    # print(f"Using subset of {len(subset_questions)} questions out of {num_questions}")
    # print(f"PMIDs used in the relevant data set include: {all_relevant_pmids}")
    # print(fetch_abstract("12345678"))
    print(f"File saved successfully? {export_file_path.exists()}")
    print(f"{num_records} created.")
    print(f"Elapsed time: {elapsed:.2f} seconds.")