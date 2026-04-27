import pandas as pd
from retriever import RetrieveDocs

def eval_results(test_csv_path, top_k=5, sm_sz=100):
    df = pd.read_csv(test_csv_path).fillna("")
    if sm_sz:
        df = df.sample(min(sm_sz, len(df)), random_state=42)
    mdl_rtr = RetrieveDocs()
    top_crt = 0
    recall_k = 0
    rec_rks = []
    for _, row in df.iterrows():
        qry = row["question"]
        tr_ans = row["answer"]
        rslts = mdl_rtr.search(qry, top_k=top_k)
        rtr_ansrs = [r["answer"] for r in rslts]
        if len(rtr_ansrs) > 0 and rtr_ansrs[0] == tr_ans:
            top_crt += 1
        is_prsnt = False
        for rank, answer in enumerate(rtr_ansrs, start=1):
            if answer == tr_ans:
                recall_k += 1
                rec_rks.append(1 / rank)
                is_prsnt = True
                break
        if not is_prsnt:
            rec_rks.append(0)
    total = len(df)
    print("Total test questions:", total)
    print("Accuracy:", top_crt / total)
    print(f"Recall@{top_k}:", recall_k / total)
    print("MRR:", sum(rec_rks) / total)

if __name__ == "__main__":
    eval_results(
        test_csv_path="/content/drive/MyDrive/ML/data/medical_rag_dataset.csv",
        top_k=5,
        sms=200
    )