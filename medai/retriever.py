import pickle
import faiss
from sentence_transformers import SentenceTransformer

class RetrieveDocs:
    def __init__(
        self,
        id_loc="data/vector_db/medical_faiss.index",
        meta_loc="data/vector_db/medical_metadata.pkl"
    ):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(id_loc)
        with open(meta_loc, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, top_k=5):
        em_query = self.model.encode([query], convert_to_numpy=True)
        vec_dist, idx_arr = self.index.search(em_query, top_k)
        results = []
        for dst, idx in zip(vec_dist[0], idx_arr[0]):
            gh = self.metadata[idx]
            results.append({
                "question": gh.get("question", ""),
                "answer": gh.get("answer", ""),
                "source": gh.get("source", ""),
                "focus_area": gh.get("focus_area", ""),
                "dataset": gh.get("dataset", ""),
                "score": float(dst)
            })
        return results