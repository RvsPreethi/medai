import os
from openai import OpenAI
from retriever import RetrieveDocs

class ImplRAG:
    def __init__(self):
        self.retriever = RetrieveDocs()
        self.client = OpenAI(
            api_key='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'
        )

    def create_prompt(self, question, retrieved_docs):
        context = ""
        for i, doc in enumerate(retrieved_docs, start=1):
            context += f"""Source {i}: {doc['dataset']} - {doc['source']} Medical Information:
{doc['answer']}"""

        prompt = f"""You are MedAI, an evidence-based medical assistant.

            Rules:
            1. Answer only using the provided medical context.
            2. Do not hallucinate or invent medical facts.
            3. If information is insufficient, say so clearly.
            4. Add:
            "This is not a medical diagnosis. Please consult a licensed healthcare professional."

            User Question:
            {question}

            Retrieved Medical Context:
            {context}

            Final Answer:
            """

        return prompt

    def generate_answer(self, question):
        retrieved_docs = self.retriever.search(question, top_k=5)
        prompt = self.create_prompt(question, retrieved_docs)
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }
if __name__ == "__main__":
    rag = ImplRAG()
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