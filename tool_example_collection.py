import langchain.embeddings as le
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

class ToolExamplesCollection:

    def __init__(self, qdrant_client: QdrantClient, embedder: le.OpenAIEmbeddings | le.HuggingFaceEmbeddings, vector_size: int):
        self.__qdrant_client = qdrant_client
        self.__embedder = embedder
        self.__vector_size = vector_size

    def create_examples_collection(self, tool_name: str, examples: list[str]) -> bool:
        try:
            self.__qdrant_client.get_collection(tool_name)
        except ValueError:
        
            # Create new tool examples collection
            self.__qdrant_client.recreate_collection(
                collection_name=tool_name,
                vectors_config=models.VectorParams(
                    size=self.__vector_size,
                    distance=models.Distance.COSINE
                )
            )

            # Add examples to collection
            for idx, example in enumerate(examples):
                self.add_example(idx, tool_name, example)
            
            return True

        return False
        
    def add_example(self, idx: int, tool_name: str, example: str):
        # Insert examples vectors into tool examples collection
        embedded_example = self.__embedder.embed_query(example)
        self.__qdrant_client.upsert(
            collection_name=tool_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedded_example,
                    payload={
                        "example_text": example,
                    }
                )
                
            ]
        )

    def list_examples(self, tool_name: str) -> list[tuple[list,str]]:
        vectors = []
        offset = 0
        while True:
            examples, offset = self.__qdrant_client.scroll(
                collection_name=tool_name,
                with_payload=True,
                with_vectors=True,
                offset=offset
            )

            # extract vectors from result
            for v in examples:
                vectors.append((v.vector,v.payload['example_text']))
            
            if offset is None:
                break

        return vectors

    def least_similar_examples(self, tool_name: str, n_elements: int = 1) -> list[tuple[float, list, str]]:
        hits = self.__qdrant_client.search(
            collection_name=tool_name,
            query_vector= self.centroid(tool_name),
            limit=self.examples_count(tool_name),
            with_vectors=True
        )

        res = []
        for idx, elem in enumerate(hits[::-1]):

            if idx >= n_elements:
                break

            res.append((elem.score, elem.vector, elem.payload['example_text']))

        return res

    def centroid(self, tool_name: str) -> int:
        examples = self.list_examples(tool_name)
        vectors = map(lambda x: x[0], examples)
        return self.__get_centroid(list(vectors))

    def examples_count(self, tool_name: str) -> int:
        collection_info = self.__qdrant_client.get_collection(tool_name)
        return collection_info.points_count

    def __get_centroid(self, vectors: list = None) -> int:
        return np.mean(vectors, axis=0).tolist()
