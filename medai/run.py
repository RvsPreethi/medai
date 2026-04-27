import preprocess as prs
import build_index as bld_idx
import retriever as rtr
import rag as rag_mdl



#all files locations
md_file_loc = "data/medquad.csv"
pqaa_file_loc = "data/ori_pqaa.json"
op_loc = "data/medical_rag_dataset.csv"
index_loc = "data/vector_db/medical_faiss.index"
metadata_loc = "data/vector_db/medical_metadata.pkl"

'''
In this file we run each step in the pipeline to generate the RAG model and then the question and answers are generated using the LLM
'''

#prs.generate_datas(md_file_loc, pqaa_file_loc, op_loc)
#bld_idx.build_faiss_index(op_loc, index_loc, metadata_loc)
#rtr.RetrieveDocs(index_loc, metadata_loc)
rag_mdl.ImplRAG()

rag = rag_mdl.ImplRAG()
while True:
    question = input("\nAsk a medical question (or type exit): ")
    if question.lower() == "exit":
        break
    result = rag.generate_answer(question)
    print("\nMedAI Answer:\n")
    print(result["answer"])
    print("\nRetrieved results:\n")
    for source in result["sources"]:
        print(
            f"- {source['dataset']} | "
            f"{source['focus_area']} | "
            f"{source['source']}"
        )