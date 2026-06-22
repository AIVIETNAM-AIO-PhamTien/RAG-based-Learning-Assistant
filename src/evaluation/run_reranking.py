from src.retriever import Retriever


if __name__ == "__main__":
    query = input("Query: ")
    for chunk in Retriever().search(query):
        print(f"{chunk.score:.3f} | {chunk.source} p.{chunk.page} | {chunk.text[:160]}")
