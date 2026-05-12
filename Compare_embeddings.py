from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def main():

    # Load embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Convert words to vectors
    apple_vector = embedding_function.embed_query("apple")
    iphone_vector = embedding_function.embed_query("iphone")

    # Print vector info
    print(f"Vector length: {len(apple_vector)}")

    # Calculate similarity
    similarity = cosine_similarity(
        [apple_vector],
        [iphone_vector]
    )

    print(f"Similarity between apple and iphone: {similarity[0][0]}")


if __name__ == "__main__":
    main()