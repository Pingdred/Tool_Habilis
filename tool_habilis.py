import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class THabilis:

    _qdrant_client = None
    _vector_size = 0
    _embedder = None
    _tools_colection_name = "tools_info"
    tools_count = 0

    def __init__(self, qdrant_client: QdrantClient, embedder, vector_size: int):
        self._qdrant_client = qdrant_client
        self._vector_size = vector_size
        self._embedder = embedder

        if not self.collection_exists(self._tools_colection_name):
            print("CREAZIONE TOOL COLLECTION")
            self.__recreate_info_collection()
        
        collection_info = self._qdrant_client.get_collection(self._tools_colection_name)
        self.tools_count = collection_info.points_count
        print(self.tools_count)


    def load(self, tools: list):
        print("Loading tools...")
        for t in tools:
            print(f"\t{t['name']}")
            self.__recreate_tool_collection(t)

    def select_by_mean(self, query: str, limit: int = 1) -> str:
        hits = self._qdrant_client.search(
            collection_name=self._tools_colection_name,
            query_vector=('mean_point',self._embedder.embed_query(query)),
            limit=limit
        )

        for elem in hits:
            print(f"MEAN VECTOR:\t {elem.payload['name']} \t {elem.score}")

    def select_by_description(self, query: str,  limit: int = 1) -> str:
        hits = self._qdrant_client.search(
            collection_name=self._tools_colection_name,
            query_vector=('description',self._embedder.embed_query(query)),
            limit=limit
        )

        for elem in hits:
            print(f"DESCR VECTOR\t {elem.payload['name']} \t {elem.score}")

    def collection_exists(self, collection_name: str) -> bool:
        # Check that the given collection exists
        for c in self._qdrant_client.get_collections().collections:
            if collection_name == c.name:
                return True

        return False

    def print_tools_collection(self):
        # Print tool_info collection
        tools = self._qdrant_client.scroll(
                collection_name='tools_info',
                with_vectors=False,
                with_payload=True,
                limit=self.tools_count
            )
        
        for v in tools[0]:
            print("-"*10, f"{v.payload['name'].upper()}", "-"*10)
            print(f"Description: {v.payload['description']}")
            print("Arguments:")
            for arg in v.payload['arguments'].items():
                print(f"\t{arg[0]}: {arg[1]}")

    def __recreate_info_collection(self):
        # Create new tools collection
        self._qdrant_client.recreate_collection(
                collection_name=self._tools_colection_name,
                vectors_config={
                    "description": models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE
                    ),
                    "mean_point": models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE
                    )
                }
            )

    def __recreate_tool_collection(self, tool: dict):
        # Create new tool examples collection
        self._qdrant_client.recreate_collection(
            collection_name=tool["name"],
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE
            )
        )

        # Add example to collection
        for idx, example in enumerate(tool["examples"]):
            self.__add_example(idx,tool["name"], example)

        # Add tool to available tools
        self.__add_tool(tool)

    def __add_example(self, idx: int, tool_name: str, example: str):
        # Insert examples vectors into tool examples collection
        self._qdrant_client.upsert(
            collection_name=tool_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=self._embedder.embed_query(example),
                    payload={
                        "name": tool_name,
                        "example_text": example,
                    }
                )
                
            ]
        )

    def __add_tool(self, tool: dict):

        mean_vector = self.__mean_vector(tool["name"])

        self._qdrant_client.upsert(
            collection_name=self._tools_colection_name,
            points=[
                models.PointStruct(
                    id=self.tools_count,
                    vector={
                        "description": self._embedder.embed_query(tool["description"]),
                        "mean_point": mean_vector
                    },
                    payload={
                        "name": tool["name"],
                        "description": tool["description"],
                        "arguments": tool["arguments"]
                        #"variance": self.__variance(tool["name"])
                    }
                )
            ]
        )
        self.tools_count += 1
        print("ADDED TO TOOL COLLECTION")

    def _get_all_examples(self, collection: str) -> list:
        examples = self._qdrant_client.scroll(
            collection_name=collection,
            with_vectors=True,
            with_payload=False
        )

        # extract vectors from result
        vectors = []
        for v in examples[0]:
            vectors.append(v.vector)

        return vectors

    def __mean_vector(self, collection: str) -> list:
        vectors = self._get_all_examples(collection)
        return np.mean(vectors, axis=0).tolist()

    # def __variance(self, collection: str) -> float:
    #     vectors = self._get_all_examples(collection)
    #     return np.var(vectors)