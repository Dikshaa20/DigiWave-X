import chromadb

# 1. Create a database that lives only in memory (nothing saved to disk yet)
client = chromadb.Client()

# 2. Create a "collection" — think of it like a table, but for text+meaning
collection = client.create_collection(name="test_collection")

# 3. Add three sentences. ChromaDB will automatically convert each one
#    into a number-list (embedding) behind the scenes.
collection.add(
    documents=[
        "AMI encoding alternates the polarity of each mark.",
        "Manchester encoding has a transition in the middle of every bit.",
        "The capital of France is Paris.",
    ],
    ids=["doc1", "doc2", "doc3"],
)

# 4. Search: which of those 3 sentences is closest in MEANING to this query?
results = collection.query(
    query_texts=["How does AMI handle bit polarity?"],
    n_results=2,
)

print(results["documents"])
print(results["distances"])