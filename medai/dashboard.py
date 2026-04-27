import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from retriever import RetrieveDocs


def get_medical_data():
    return pd.read_csv("data/medical_rag_dataset.csv")


def generate_header():
    st.set_page_config(page_title="MedAI Application Dashboard", layout="wide")
    st.title("MedAI Application Dashboard")

#Show the dataset in samples.
#In here we show all the data related content and information
def show_dataset_overview(df):
    st.header("Dataset Overview")
    records_col, data_col, focus = st.columns(3)
    records_col.metric("Total Records", len(df))
    data_col.metric("Datasets Used", df["dataset"].nunique())
    focus.metric("Research Areas", df["focus_area"].nunique())
    st.subheader("Sample Data")
    st.dataframe(df[["question", "answer", "source", "focus_area", "dataset"]].head(20))


def get_data_details(df):
    st.header("Dataset Details")
    dataset_counts = df["dataset"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(dataset_counts.index, dataset_counts.values)
    ax.set_title("Records in the dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)


def research_name(df):
    st.header("Top Medical research Areas")
    focus_counts = df["focus_area"].value_counts().head(10)
    fig, ax = plt.subplots()
    ax.barh(focus_counts.index, focus_counts.values)
    ax.set_title("Top 10 Focus Areas")
    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Focus Area")
    st.pyplot(fig)


def show_retrieved_docs(df):
    st.header("Retrieved Documents")
    df["answer_length"] = df["answer"].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots()
    ax.hist(df["answer_length"], bins=30)
    ax.set_title("Answer Length Distribution")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def doc_Scores(result_df):
    scores = result_df["score"].tolist()
    labels = [f"Result {i + 1}" for i in range(len(scores))]
    fig, ax = plt.subplots()
    ax.bar(labels, scores)
    ax.set_title("FAISS Distance Scores")
    ax.set_xlabel("Retrieved Documents")
    ax.set_ylabel("Document Distance Score")
    st.pyplot(fig)


def get_results_arr(results):
    st.subheader("Top Retrieved Medical Documents")
    for i, result in enumerate(results, start=1):
        st.markdown(f"Result {i}")
        st.write("Question:", result["question"])
        st.write("Answer:", result["answer"])
        st.write("Dataset:", result["dataset"])
        st.write("Focus Area:", result["focus_area"])
        st.write("Source:", result["source"])
        st.write("Score:", result["score"])


def rag_impl():
    st.header("Document Retrieval")
    query = st.text_input("Enter a medical question:")
    if query:
        retriever = RetrieveDocs()
        results = retriever.search(query, top_k=5)
        st.subheader("Retrieved Results")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df[["question", "dataset", "focus_area", "source", "score"]])
        doc_Scores(result_df)
        get_results_arr(results)

def implement_dashboard():
    generate_header()
    df = get_medical_data()
    show_dataset_overview(df)
    get_data_details(df)
    research_name(df)
    show_retrieved_docs(df)
    rag_impl()



implement_dashboard()