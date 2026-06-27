from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class VectorDBService:
    def __init__(self):
        print("Initializing local In-Memory Qdrant Client...")
        self.client = QdrantClient(location=":memory:")
        self.collection_name = "codeBASE_chunks"
        print("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.vector_size = 384  
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            pass

    def upsert_code_chunks(self, chunks: list[dict]):
        if not chunks:
            return

        print(f"Generating embeddings for {len(chunks)} code chunks...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,  
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "file_path": chunk["metadata"]["file_path"],
                        "start_line": chunk["metadata"]["start_line"],
                        "end_line": chunk["metadata"]["end_line"],
                        "context": chunk["metadata"]["context"],
                        "language": chunk["metadata"]["language"]
                    }
                )
            )

        print(f"Upserting vectors into In-Memory Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    def search_similar_code(self, query: str, limit: int = 5) -> list[dict]:
        try:
            query_vector = self.model.encode(query).tolist()
            if hasattr(self.client, "query_points"):
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit
                ).points
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit
                )

            retrieved_chunks = []
            for hit in search_results:
                retrieved_chunks.append({
                    "content": hit.payload["content"],
                    "score": getattr(hit, "score", 1.0),
                    "metadata": {
                        "file_path": hit.payload["file_path"],
                        "start_line": hit.payload["start_line"],
                        "end_line": hit.payload["end_line"],
                        "context": hit.payload["context"],
                        "language": hit.payload["language"]
                    }
                })
                
            return retrieved_chunks

        except Exception as e:
            print(f"Error during vector DB search execution: {str(e)}")
            return []