import chromadb
c = chromadb.PersistentClient(path="/opt/car-ticket-chatbot/data/chroma")
coll = c.get_collection("rag_v2")
print("Collection metadata:", coll.metadata)

# Check ChromaDB version
import importlib
print("ChromaDB version:", chromadb.__version__)

# Check distance metric used
print("\nCollection name:", coll.name)

# Get a sample to check id format
sample = coll.get(where={"type": "ticket"}, limit=3)
print("\nSample ticket IDs:", sample["ids"])
print("Sample ticket metas:", sample["metadatas"])
