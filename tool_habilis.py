import langchain.embeddings as le
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

import tool_example_collection as txc 


class ToolHabilis:

    def __init__(self, qdrant_client: QdrantClient, embedder: le.OpenAIEmbeddings | le.HuggingFaceEmbeddings, vector_size: int, tools_collection_name: str = "tools_info"):
        self.__qdrant_client = qdrant_client
        self.__vector_size = vector_size
        self.__tools_collection_name = tools_collection_name
        self.__embedder = embedder
        self.__tool_example_collection = txc.ToolExamplesCollection(self.__qdrant_client, embedder, self.__vector_size)

        try:
            self.__qdrant_client.get_collection(self.__tools_collection_name)
        except ValueError:
            self.__create_info_collection()

    def add_tool(self, tool_name: str, tool_descr: str, examples: list[str], tool_args: list[tuple]) -> bool:

        if not self.__tool_example_collection.create_examples_collection(tool_name, examples):
            return False

        centroid = self.__tool_example_collection.centroid(tool_name)
        similarity, vector, text = self.__tool_example_collection.least_similar_examples(tool_name)[0]

        self.__qdrant_client.upsert(
            collection_name=self.__tools_collection_name,
            points=[
                models.PointStruct(
                    id=self.tools_count(),
                    vector={
                        "description": self.__embedder.embed_query(tool_descr),
                        "centroid": centroid
                    },
                    payload={
                        "name": tool_name,
                        "description": tool_descr,
                        "arguments": tool_args,
                        "least_similar_example": {
                            "vector": vector,
                            "similarity": similarity,
                            "text": text
                        },
                        "margin": 0.01,
                        "similarities_rms": 0,
                        "similarities_variance": 0
                    }
                )
            ]
        )

        return True

    def list_tools(self) -> list:
        tools = self.__qdrant_client.scroll(
            collection_name=self.__tools_collection_name,
            with_vectors=True
        )
        
        return tools[0]

    def select_by_centroid_sim(self, query: str, limit: int = 1, limit_similarity: bool = True) -> list[tuple[str,float]]:
        hits = self.__qdrant_client.search(
            collection_name=self.__tools_collection_name,
            query_vector=('centroid',self.__embedder.embed_query(query)),
            limit=limit
        )

        res = []
        for elem in hits:
            margin = elem.payload["margin"]

            if limit_similarity:
                if (elem.score+margin) >= elem.payload['least_similar_example']['similarity']:
                    res.append((elem.payload['name'], elem.score))
            else:
                res.append((elem.payload['name'], elem.score))            
        
        return res
    
    def select_by_description_sim(self, query: str,  limit: int = 1) -> str:
        hits = self.__qdrant_client.search(
            collection_name=self.__tools_collection_name,
            query_vector=('description',self.__embedder.embed_query(query)),
            limit=limit
        )
        
        res = map(lambda elem: (elem.payload['name'], elem.score), hits)
        return list(res)

    def print_tools_collection(self):

        # Print tools_info collection
        tools = self.list_tools()
        
        for v in tools[0]:
            print("-"*20, f"{v.payload['name'].upper()}", "-"*20)
            print(f"Description: {v.payload['description']}")
            print("Arguments:")
            for arg in v.payload['arguments'].items():
                print(f"\t{arg[0]}: {arg[1]}")
            print()

    def tools_count(self) -> int:
        collection_info = self.__qdrant_client.get_collection(self.__tools_collection_name)
        return collection_info.points_count

    def check_tools_similarity(self, min_similarity: float = 0):
        tools = self.list_tools()
        collition = []
        for index, elem_1 in enumerate(tools):
            for elem_2 in tools[index+1:]:
                similarity = self.__collide(elem_1.payload['name'], elem_2.payload['name'])
                if similarity >= min_similarity:
                    collition.append((elem_1.payload['name'], elem_2.payload['name'], similarity))
        
        return collition

    def __create_info_collection(self):
        # Create new tools collection
        self.__qdrant_client.recreate_collection(
                collection_name=self.__tools_collection_name,
                vectors_config={
                    "description": models.VectorParams(
                        size=self.__vector_size,
                        distance=models.Distance.COSINE
                    ),
                    "centroid": models.VectorParams(
                        size=self.__vector_size,
                        distance=models.Distance.COSINE
                    )
                }
            )
    
    def __get_tool(self, tool_name: str):
        tool = self.__qdrant_client.scroll(
            collection_name=self.__tools_collection_name,
            with_vectors=True,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="name",
                        match=models.MatchValue(value=tool_name),
                    ),
                ]
            ),
        )

        return tool[0][0]

    def __collide(self, t_1: str, t_2: str) -> float:
        t_1 = self.__get_tool(t_1)
        t_2 = self.__get_tool(t_2)

        t_1_centroid = np.array(t_1.vector['centroid'])
        t_2_centroid = np.array(t_2.vector['centroid'])
        centroid_similarity = np.dot(t_1_centroid, t_2_centroid)

        #first_tool_description = np.array(first_tool.vector['centroid'])
        #second_tool_description = np.array(second_tool.vector['centroid'])
        #description_similarity = np.dot(first_tool_description, second_tool_description)

        #if centroid_similarity 

        return centroid_similarity