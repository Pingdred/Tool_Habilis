import json
import sys

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from qdrant_client import QdrantClient

from tool_habilis import ToolHabilis
import api_keys


QDRANT_DB_PATH = "./qdb_plugins"
SIMILAR_TOOL_THRESHOLD = 0.3
# EMBEDDER_VECTOR_SIZE = 1536 # OpenAI
EMBEDDER_VECTOR_SIZE = 768 # HuggingFace

if __name__ == "__main__":

    client = QdrantClient(path=QDRANT_DB_PATH)

    embedder = HuggingFaceEmbeddings()
    # embedder = OpenAIEmbeddings(openai_api_key=api_keys.OPENAI_KEY)

    tool_chooser = ToolHabilis(client, embedder, EMBEDDER_VECTOR_SIZE)

    if len(sys.argv) == 2:
        print("Opening tools file.")
        with open(sys.argv[1], 'r') as tools_file:
            tools = json.load(tools_file)

        tools = tools['tools_list']
        
        print("Loading tools...")
        for t in tools:
            res = tool_chooser.add_tool(t['name'], t['description'], t['examples'], t['arguments'])
            print(f"\t{t['name']}: {'OK' if res else 'ALREADY EXISTS'}")

    # tool_chooser.print_tools_collection()
    print("\nAvailable tools: ", tool_chooser.tools_count())

    similar_tools = tool_chooser.check_tools_collisions()
    if len(similar_tools) > 0:
        print(f"There are {len(similar_tools)} tools collisions:")
        for c in similar_tools:
            print(f"\t{c[0]} -> {c[1]}: {c[2]}")
    else:
        print("No tools collison")


    while True:
        print("-"*30)
        query = input("Query (q to exit): ")

        if query == 'q' or query == '':
            break

        midpoint_sim_hit= tool_chooser.select_by_midpoint_sim(query)
        midpoint_sim_nearest= tool_chooser.select_by_midpoint_sim(query=query, limit_similarity=False)
        description_sim_hit = tool_chooser.select_by_description_sim(query)

        print("HITTED TOOL")
        for elem in midpoint_sim_hit:
            print(f"{elem[0]}:")
            print(f"\tQuery sim: {round(elem[1],4)}")
            print(f"\tRadius: {round(elem[2],4)}")
            print(f"\tMargin: {round(elem[3],4)}\n")

        print("NEAREST BY MIDPOINT")
        for elem in midpoint_sim_nearest:
            print(f"{elem[0]}:")
            print(f"\tQuery sim: {round(elem[1],4)}")
            print(f"\tRadius: {round(elem[2],4)}")
            print(f"\tMargin: {round(elem[3],4)}\n")

        print("NEAREST BY DESCRIPTION ")
        for elem in description_sim_hit:
            print(f"{elem[0]}:")
            print(f"Query sim: {round(elem[1],4)}")

        print()
        
