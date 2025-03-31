from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryResponse

QDRANT_URL = "http://localhost:6333"


class QdrantConnection:
    """Class to manage connection to Qdrant server."""

    def __init__(self):
        self.client = QdrantClient(QDRANT_URL)

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def insert(self, collection_name: str, points: list[PointStruct]):
        self.client.upload_points(collection_name, points, batch_size=1000)

    def search(self, collection_name: str, query: list[float]) -> QueryResponse:
        return self.client.query_points(collection_name, query)

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)

