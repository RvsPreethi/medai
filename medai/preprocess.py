import os
import json
import pandas as pd
#In this step we create the database for the medquad and pubmedqa data. LOad the data from relative input data files and then use them for analysis
#get data from medquad and add to main data file
def add_medqa_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "question": "question",
        "answer": "answer",
        "source": "source",
        "focus_area": "focus_area"
    })
    df["context"] = df["answer"]
    df["dataset"] = "MedQuAD"
    df = df[["question", "answer", "context", "source", "focus_area", "dataset"]]
    return df

#get data from pubmed and add to main data file
def load_pubmedqa(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for qId, item in data.items():
        question = item.get("QUESTION", "")
        decs = item.get("final_decision", "")
        l_a = item.get("LONG_ANSWER", "")
        contx_arr = item.get("CONTEXTS", [])
        context_text = " ".join(contx_arr) if isinstance(contx_arr, list) else str(contx_arr)
        answer = l_a if l_a else decs
        rows.append({
            "question": question,
            "answer": answer,
            "context": context_text,
            "source": f"PubMedQA ID: {qId}",
            "focus_area": "Biomedical Research",
            "dataset": "PubMedQA"
        })

    df = pd.DataFrame(rows)
    return df

#merge data from medquad and pubmedqa
def merge_data(df):
    #print(df.head())
    df = df.fillna("")
    df["question"] = df["question"].astype(str).str.strip()
    #print(df["question"])
    df["answer"] = df["answer"].astype(str).str.strip()
    #print(df["answer"])
    df["context"] = df["context"].astype(str).str.strip()
    #print(df["context"])
    df["source"] = df["source"].astype(str).str.strip()
    #print(df["source"])
    df["focus_area"] = df["focus_area"].astype(str).str.strip()
    #print(df["focus_area"])
    df = df[df["question"] != ""]
    df = df[df["answer"] != ""]
    df = df.drop_duplicates(subset=["question", "answer"])
    df = df.reset_index(drop=True)
    #print(df.shape)
    return df


def create_db(df):
    df["rag_text"] = (
        "Question: " + df["question"] + "\n"
        "Answer: " + df["answer"] + "\n"
        "Context: " + df["context"] + "\n"
        "Source: " + df["source"] + "\n"
        "Focus Area: " + df["focus_area"]
    )
    #print(df.head())
    return df


def generate_datas(medquad_path, pubmedqa_path, output_path):
    m1 = add_medqa_data(medquad_path)
    p1 = load_pubmedqa(pubmedqa_path)
    join_m_p = pd.concat([m1, p1], ignore_index=True)
    join_m_p = merge_data(join_m_p)
    join_m_p = create_db(join_m_p)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    join_m_p.to_csv(output_path, index=False)
    print("Generated data for medquad and pubmedqa")
    print("Total records:", len(join_m_p))
    print("Saved to:", output_path)

if __name__ == "__main__":
    medquad_path = "data/medquad.csv"
    pubmedqa_path = "data/ori_pqaa.json"
    output_path = "processed/medical_rag_dataset.csv"
    generate_datas(medquad_path, pubmedqa_path, output_path)