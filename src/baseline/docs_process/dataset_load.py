from datasets import load_dataset, Dataset
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm

def load_eval_data(dataset_name :str = "legal" ) :
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    if dataset_name  ==  "legal" :
        dataset = load_dataset("tuananh18/Eval-RAG-Vietnamese", 'legal-data', cache_dir= "./")

    elif dataset_name == "viQuAD" : 
        pass 
    elif dataset_name == "expert" : 
        pass 
    
    return dataset

def preprocess() :
    pass 

def document_transform(ds : Dataset) -> LangchainDocument:
    docs = [
    LangchainDocument(page_content=row["context"])
    for  row in tqdm(ds['train'])
]
    return docs