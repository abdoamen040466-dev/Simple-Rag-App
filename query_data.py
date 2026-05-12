import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"
load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Input a question.
    query_text = input("Ask a question: ")
    # Prepare the DB.
    embedding_function = embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    for doc, score in results:
        print(f"\nSimilarity Score: {score}")

    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\nSearching database...\n")

    model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0)

    response = model.invoke(prompt)
    response_text = response.content

    formatted_response = f"Response: {response_text}"
    print(formatted_response)


if __name__ == "__main__":
    main()