import langchain.embeddings as le
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class ToolHabilis:

    def __init__(self, qdrant_client: QdrantClient, embedder: le.OpenAIEmbeddings | le.HuggingFaceEmbeddings, vector_size: int, tools_collection_name: str = "tools_info"):
        self.__qdrant_client = qdrant_client
        self.__vector_size = vector_size
        self.__tools_collection_name = tools_collection_name
        self.__embedder = embedder

        try:
            self.__qdrant_client.get_collection(self.__tools_collection_name)
        except ValueError:
            self.__create_info_collection()

    def add_tool(self, tool_name: str, tool_descr: str, examples: list[str], tool_args: list[tuple]) -> bool:

        if self.__get_tool(tool_name) is not None:
            return False

        less_similar_example = self.__less_similar_examples(examples)[0]
        midpoint = np.mean( [less_similar_example[0][1], less_similar_example[1][1]], axis=0).tolist()
        radius = np.dot(less_similar_example[0][1], midpoint)
        
        self.__qdrant_client.upsert(
            collection_name=self.__tools_collection_name,
            points=[
                models.PointStruct(
                    id=self.tools_count(),
                    vector={
                        "description": self.__embedder.embed_query(tool_descr),
                        "midpoint": midpoint
                    },
                    payload={
                        "name": tool_name,
                        "description": tool_descr,
                        "arguments": tool_args,
                        "radius": radius,
                        "less_similar": less_similar_example[0][1],
                        "margin": 0.01,
                        "similarities_rms": 0,
                        "similarities_variance": 0
                    }
                )
            ]
        )

        return True

    def list_tools(self) -> list:
        vectors = []
        offset = 0
        while True:
            tools, offset= self.__qdrant_client.scroll(
                collection_name=self.__tools_collection_name,
                with_vectors=True,
                limit=self.tools_count(),
                offset=offset
            )

             # extract vectors from result
            for t in tools:
                vectors.append(t)
            
            if offset is None:
                break
        
        return vectors

    def select_by_midpoint_sim(self, query: str, limit: int = 1, limit_similarity: bool = True) -> list[tuple[str,float]]:
        hits = self.__qdrant_client.search(
            collection_name=self.__tools_collection_name,
            query_vector=('midpoint',self.__embedder.embed_query(query)),
            limit=limit
        )

        res = []
        for elem in hits:
            margin = elem.payload["margin"]

            tmp = (elem.payload['name'], elem.score, elem.payload['radius'], elem.payload['margin'])

            if not limit_similarity or (elem.score >= elem.payload['radius'] + margin):
                res.append(tmp) 
        
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

    def check_tools_collisions(self):
        tools = self.list_tools()
        collisions = []
        for index, elem_1 in enumerate(tools):
            for elem_2 in tools[index+1:]:
                similarity, collision = self.__collide(elem_1.payload['name'], elem_2.payload['name'])
                
                if collision:
                    collisions.append((elem_1.payload['name'], elem_2.payload['name'], similarity))
        
        return collisions

    def __less_similar_examples(self, examples: list, limit: int = 1) -> list[tuple[tuple[str,list],tuple[str,list], float]]:
        less_similar = []

        for idx, e_1 in enumerate(examples):
            for e_2 in examples[(idx+1):]:
                e_1_embedded = self.__embedder.embed_query(e_1)
                e_2_embedded = self.__embedder.embed_query(e_2)

                similarity = np.dot( e_1_embedded, e_2_embedded)
                less_similar.append(((e_1, e_1_embedded), (e_2, e_2_embedded), similarity))

        less_similar = sorted(less_similar, key=lambda x: x[2])

        return less_similar [:limit]

    def __create_info_collection(self):
        # Create new tools collection
        self.__qdrant_client.recreate_collection(
                collection_name=self.__tools_collection_name,
                vectors_config={
                    "description": models.VectorParams(
                        size=self.__vector_size,
                        distance=models.Distance.COSINE
                    ),
                    "midpoint": models.VectorParams(
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

        if len(tool[0]) == 0:
            return None

        return tool[0][0]

    def __collide(self, t_1: str, t_2: str) -> tuple[float,bool]:
        t_1 = self.__get_tool(t_1)
        t_2 = self.__get_tool(t_2)

        t_1_midpoint = np.array(t_1.vector['midpoint'])
        t_2_midpoint = np.array(t_2.vector['midpoint'])
        midpoint_similarity = np.dot(t_1_midpoint, t_2_midpoint)

        collision = midpoint_similarity >= (t_1.payload['radius'] + t_1.payload['margin']) + (t_1.payload['radius'] + t_1.payload['margin'])

        return (midpoint_similarity, collision)