**MedAI - Retrieval-Augmented Medical Question Answering System**
**Overview**

MedAI is a Retrieval-Augmented Generation (RAG) based medical question answering system designed to provide evidence-based healthcare responses. The project combines semantic search, vector databases, and Large Language Models to retrieve relevant medical information and generate context-aware answers.

The system uses trusted biomedical datasets including MedQuAD and PubMedQA to reduce hallucinations and improve factual accuracy in generated responses.

**Features**
  Medical question answering using RAG architecture
  Semantic embedding generation using SentenceTransformers
  FAISS vector database for fast similarity search
  OpenAI-based answer generation
  Streamlit dashboard for visualization and analytics
  Retrieval evaluation using Top-1 Accuracy, Recall@5, and MRR
  Datasets
  MedQuAD

[https://github.com/abachaa/MedQuAD](url)

PubMedQA

[https://pubmedqa.github.io/](url)

**Technologies Used**
  Python 3.11
  SentenceTransformers
  FAISS
  OpenAI API
  Pandas
  Streamlit
  Matplotlib

  **pip install pandas sentence-transformers faiss-cpu openai streamlit matplotlib scikit-learn**

**  GitHub Repository**

  https://github.com/RvsPreethi/medai

**Commands to run: **
python preprocess.py

python build_vector_db.py

python evaluate_retriever.py

python retriever.py

set OPENAI_API_KEY=your_api_key

python rag_pipeline.py

# Step 8: Launch Streamlit dashboard
streamlit run dashboard.py
