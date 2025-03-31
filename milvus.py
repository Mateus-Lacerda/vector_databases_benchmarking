from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"


class MilvusConnection:
    """Class for managing Milvus connections"""

    def __init__(self):
        self.client = MilvusClient(MILVUS_URI)

    def create_collection(self, collection_name):
        """Create a collection"""
        dimension = 384 
        self.client.create_collection(collection_name, dimension)
        res = self.client.get_load_state(collection_name)
        print(f"Collection {collection_name} load state: {res}")
        desc = self.client.describe_collection(collection_name)
        print(f"Collection {collection_name} description: {desc}")

    def insert(self, collection_name, vectors):
        """Insert vectors"""
        return self.client.insert(collection_name, vectors)

    def search(self, collection_name, vectors, top_k=5):
        """Search vectors"""
        search_params = {"metric_type": "COSINE"}
        return self.client.search(
            collection_name=collection_name,
            anns_field="vector",
            data=vectors,
            limit=top_k,
            search_params=search_params,
            output_fields=["key"]
        )

    def delete_collection(self, collection_name):
        """Delete a collection"""
        return self.client.drop_collection(collection_name)

