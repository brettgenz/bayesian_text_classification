import os
import dotenv
from pathlib import Path
import openai
import tiktoken
from Bio import Entrez

env_file_path = (Path(__file__).parent / "../.env").resolve()
dotenv.load_dotenv(env_file_path)

# define email for use with fetch_abstract()
Entrez.email = "brettgenz@gmail.com"

# establish OpenAI session
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# define functions
def fetch_abstract(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)
    try:
        abstract = records['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        return abstract
    except:
        return None


def truncate_text(text, max_tokens=8192, encoding_name="cl100k_base"):
    """
    Truncates the text to a maximum number of tokens using the tiktoken library.

    Example usage:

    df['truncated_text'] = df['text'].apply(lambda t: truncate_text(t, max_tokens=8192))
    """
    encoding = tiktoken.get_encoding(encoding_name=encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return(text)
    else:
        # Truncate tokens to max_tokens
        truncated_tokens = tokens[:max_tokens]
        text = encoding.decode(truncated_tokens)
    
    return text


def get_embedding(text):
    final_text = truncate_text(text)
    response = client.embeddings.create(input=final_text, model="text-embedding-ada-002")

    return response.data[0].embedding


if __name__ == '__main__':
    example_pmid = "12345678"
    print(fetch_abstract(example_pmid))
    embedding = get_embedding("What genes are associated with diabetes?")
    print(f"Embedding length: {len(embedding)}")