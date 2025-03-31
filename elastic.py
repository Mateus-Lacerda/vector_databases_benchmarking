"""Módulo de conexão com o Elasticsearch."""

import os
import logging
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", '9200'))
ELASTIC_USER = os.getenv("ELASTIC_USER", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")


class ElasticsearchConnection:
    """Classe responsável por realizar a conexão com o Elasticsearch."""

    def __init__(self) -> None:
        self.es = Elasticsearch(
            hosts=[f"https://{ES_HOST}:{ES_PORT}"],
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
            verify_certs=False,
            max_retries=30,
            retry_on_timeout=True,
            request_timeout=30,
            ssl_show_warn=False,
        )

    def create_index(self, index_name: str) -> dict:
        """Cria um índice no Elasticsearch."""
        return self.es.indices.create(index="rag_benchmark", body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "text_vector": {"type": "dense_vector", "dims": 384}
                    }
                }
            }, ignore=400)

    # def bulk_index(self, index_name: str, vectors: list[dict]) -> None:
    #     """Insere um novo documento no Elasticsearch."""
    #     batch_size = 1000
    #     for i in range(0, len(vectors), batch_size):
    #         batch = vectors[i:i + batch_size]
    #         operations = []
    #         for vector in batch:
    #             operations.extend([
    #                 {
    #                     "index": {
    #                         "_index": index_name,
    #                         "_id": vector['text']
    #                     }
    #                 },
    #                 {
    #                     "text": vector['text'],
    #                     "text_vector": vector['text_vector']
    #                 }
    #             ])
    #         self.es.bulk(
    #             operations=operations,
    #         )
    #     self.es.indices.refresh(index=index_name)
    #
    def bulk_index(self, index_name: str, batches: list[list[dict]]) -> None:
        """Insere vários documentos no Elasticsearch."""
        for batch in batches:
            self.es.bulk(operations=batch)
        self.es.indices.refresh(index=index_name)

    def search(self, query: list[float], top_k: int = 5) -> str:
        """Realiza a busca no Elasticsearch."""
        if not query:
            return ""
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                        "params": {
                            "query_vector": query
                        }
                    }
                }
            }
        }
        response = self.es.search(index="rag_benchmark", body=body)
        return response

    def delete_index(self, index_name: str) -> dict:
        """Deleta um índice no Elasticsearch."""
        return self.es.indices.delete(index=index_name, ignore=[400, 404])

