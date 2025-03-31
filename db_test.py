"""
Benchmarking the performance of different databases for storing embeddings.
"""

import json
import time
import uuid
import argparse
import subprocess
import threading
from enum import Enum
from typing import Iterable

import pandas as pd
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from databases.elastic import ElasticsearchConnection
from databases.milvus import MilvusConnection
from databases.qdrant import QdrantConnection
from plots import plot_times, plot_usages


class TestType(Enum):
    ELASTICSEARCH = 'elasticsearch'
    MILVUS = 'milvus'
    QDRANT = 'qdrant'
    ALL = 'all'


def start_monitoring_thread(container) -> tuple[
    threading.Thread,
    callable,
    callable
]:
    """Monitoring thread for containers usage"""
    FINISHED = False
    OPERATION = None

    def stop_callback() -> bool:
        nonlocal FINISHED
        if FINISHED:
            return False
        return True

    def set_finished():
        nonlocal FINISHED
        FINISHED = True

    def get_op():
        nonlocal OPERATION
        return OPERATION

    def set_op(op):
        nonlocal OPERATION
        OPERATION = op

    def monitor_usage():
        while stop_callback():
            stats = subprocess.run(['docker', 'stats', '--no-stream'], capture_output=True, text=True)
            lines = stats.stdout.strip().split('\n')[1:]
            with open(f'docker_stats_{container}.txt', 'a') as f:
                f.write(f"Stats for {get_op()} in {container}:\n")
                for line in lines:
                    f.write(line + '\n')

    thread = threading.Thread(target=monitor_usage)
    return thread, set_finished, set_op


def start_container(container_name: str) -> None:
    """Start a docker container"""
    subprocess.run(['docker', 'compose', 'up', '--build', '--detach', container_name], check=True)
    subprocess.run(['docker', 'ps'], check=True)


def destroy_container(container_name: str) -> None:
    """Destroy a docker container"""
    subprocess.run(['docker', 'compose', 'down', '-v'], check=True)
    subprocess.run(['docker', 'ps'], check=True)


def get_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


times = {
    'es': {},
    'milvus': {},
    'qdrant': {}
}


def save_times():
    with open('times.json', 'w') as f:
        json.dump(times, f)


def get_embedding(text):
    model = get_model()
    embedding = model.encode(text).tolist()
    return embedding


def get_text(key):
    embeddings = get_embeddings()
    index = list(embeddings.keys()).index(key)
    data = get_data()
    return data.iloc[index]['passage']


def get_data():
    data = pd.read_parquet('rag-mini-wikipedia/data/passages.parquet')
    print(data.head())
    return data


def generate_embeddings() -> None:
    data = get_data()
    embeddings = {}
    for i, row in data.iterrows():
        print(row)
        embeddings[uuid.uuid4().hex] = get_embedding(row['passage'])
    json.dump(embeddings, open('embeddings.json', 'w'))


def get_embeddings() -> dict:
    return json.load(open('embeddings.json'))


def get_query() -> list[float]:
    query_string = "What did The Legal Tender Act of 1862 establish?"
    return get_embedding(query_string)


def test_es(embeddings: dict, query: list[float]):
    # Batch formation for Elasticsearch
    batches = []
    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        batch = list(embeddings.items())[i:i + batch_size]
        ops = []
        for key, value in batch:
            ops.extend([
                {
                    'index': {
                        '_index': 'rag_benchmark',
                        '_id': key
                    }
                },
                {
                    'text': key,
                    'text_vector': value
                }
            ])
        batches.append(ops)

    start_container('es01')
    print("Waiting for Elasticsearch to start...")
    time.sleep(60)
    es_conn = ElasticsearchConnection()
    monitoring_thread, callback, set_op = start_monitoring_thread('es01')

    set_op('create_index')
    monitoring_thread.start()
    start_time = time.time()
    es_conn.create_index('rag_benchmark')
    end_time = time.time()
    times['es']['create_index'] = end_time - start_time

    set_op('insert')
    start_time = time.time()
    es_conn.bulk_index('rag_benchmark', batches)
    end_time = time.time()
    times['es']['insert'] = end_time - start_time

    set_op('search')
    start_time = time.time()
    query = get_query()
    search_results = es_conn.search(query)
    end_time = time.time()
    times['es']['search'] = end_time - start_time

    callback()
    monitoring_thread.join()
    destroy_container('es01')

    results = [
        get_text(
            hit['_source']['text']
        ) for hit in search_results['hits']['hits']
    ]
    times['es']['search_results'] = results
    print(times['es'])


def test_milvus(embeddings: dict, query: list[float]):
    # Batch formation for Milvus
    start_container('standalone')
    embeddings = [
        {
            'id': i, 'vector': value, 'key': key
        } for i, (key, value) in enumerate(embeddings.items())
    ]
    milvus_conn = MilvusConnection()
    monitoring_thread, callback, set_op = start_monitoring_thread('standalone')

    set_op('create_collection')
    monitoring_thread.start()
    start_time = time.time()
    milvus_conn.create_collection('rag_benchmark')
    end_time = time.time()
    times['milvus']['create_collection'] = end_time - start_time

    set_op('insert')
    start_time = time.time()
    milvus_conn.insert('rag_benchmark', embeddings)
    end_time = time.time()
    times['milvus']['insert'] = end_time - start_time

    set_op('search')
    start_time = time.time()
    query = get_query()
    end_time = time.time()
    search_results = milvus_conn.search('rag_benchmark', [query])
    times['milvus']['search'] = end_time - start_time

    callback()
    monitoring_thread.join()
    destroy_container('standalone')

    results = [
        get_text(hit['entity']['key']) for hit in search_results[0]
    ]
    times['milvus']['search_results'] = results
    print(times['milvus'])


# Qdrant uses an Iterable of PointStruct for inserting data
def get_points(embeddings: dict) -> Iterable[PointStruct]:
    """Convert embeddings to PointStruct for Qdrant"""
    for i, (key, value) in enumerate(embeddings.items()):
        yield PointStruct(
            id=i, vector=value, payload={'key': key}
        )


def test_qdrant(embeddings: dict, query: list[float]):
    start_container('qdrant')
    print("Waiting for Qdrant to start...")
    time.sleep(10)
    qdrant_conn = QdrantConnection()
    monitoring_thread, callback, set_op = start_monitoring_thread('qdrant')

    set_op('create_collection')
    monitoring_thread.start()
    start_time = time.time()
    qdrant_conn.create_collection('rag_benchmark')
    end_time = time.time()
    times['qdrant']['create_collection'] = end_time - start_time

    set_op('insert')
    start_time = time.time()
    qdrant_conn.insert('rag_benchmark', get_points(embeddings))
    end_time = time.time()
    times['qdrant']['insert'] = end_time - start_time

    set_op('search')
    start_time = time.time()
    query = get_query()
    search_results = qdrant_conn.search('rag_benchmark', query)
    end_time = time.time()
    times['qdrant']['search'] = end_time - start_time

    callback()
    monitoring_thread.join()
    destroy_container('qdrant')

    results = [
        get_text(hit.payload.get("key")) for hit in search_results.points
    ]

    times['qdrant']['search_results'] = results
    print(times['qdrant'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking the performance of different databases for storing embeddings')
    parser.add_argument('--prepare', action='store_true', help='Generate embeddings')
    parser.add_argument('--test', type=str, choices=list(map(str.lower, TestType._member_names_)), help='Test type')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    args = parser.parse_args()

    if args.prepare:
        generate_embeddings()
    elif args.test:
        embeddings = get_embeddings()
        query = get_query()

    if args.test == TestType.ALL.value:
        test_es(embeddings, query)
        test_milvus(embeddings, query)
        test_qdrant(embeddings, query)
        save_times()
        exit(0)
    elif args.test == TestType.ELASTICSEARCH.value:
        test_es(embeddings, query)
        save_times()
        exit(0)
    elif args.test == TestType.MILVUS.value:
        test_milvus(embeddings, query)
        save_times()
        exit(0)
    elif args.test == TestType.QDRANT.value:
        test_qdrant(embeddings, query)
        save_times()
        exit(0)

    if args.plot:
        plot_times()
        plot_usages()
        exit(0)

    parser.print_help()

