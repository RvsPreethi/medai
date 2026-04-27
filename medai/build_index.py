'''
Indexing medqa and pubmed data. We used the faiss library here for generating index
For retrieb=ving the using rag we pass this index file and then use for the answering the queries from the user.
'''
import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# In this step we generate faiss index for the created data and save it to the output path
def build_faiss_index(processed_csv_path, index_output_path, metadata_output_path):
    #get the data from the processed csv file
    df = pd.read_csv(processed_csv_path)
    df = df.fillna("")
    dt_from_file = df["rag_text"].tolist()
    #Load emebedding model and generate the embeds
    e_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model is loaded and used model is all-MiniLM-L6-v2")
    gen_e_m = e_m.encode(dt_from_file, show_progress_bar=True, convert_to_numpy=True)
    e_m_shape = gen_e_m.shape[1]
    # Now generating our FAISS index for the detection process
    print("Building FAISS index for the detection process")
    em_idx = faiss.IndexFlatL2(e_m_shape)
    em_idx.add(gen_e_m)
    # Load the indices and related metadata to the output path
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(em_idx, index_output_path)
    metadata = df.to_dict(orient="records")
    with open(metadata_output_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Vector Index is created at {index_output_path}.")
    print("Total documents in the index file is:", len(dt_from_file))
