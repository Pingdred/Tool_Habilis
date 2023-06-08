import json
import sys

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from qdrant_client import QdrantClient

from tool_habilis import ToolHabilis
import api_keys


QDRANT_DB_PATH = "./qdb_plugins"
SIMILAR_TOOL_THRESHOLD = 0.3
EMBEDDER_VECTOR_SIZE = 768


if __name__ == "__main__":

    client = QdrantClient(path=QDRANT_DB_PATH)
    embedder = HuggingFaceEmbeddings()
    #embedder = OpenAIEmbeddings(openai_api_key=api_keys.OPENAI_KEY)
    #chat = ChatOpenAI(temperature=0)
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

    #tool_chooser.print_tools_collection()
    print("Available tools: ", tool_chooser.tools_count())

    similar_tools = tool_chooser.check_tools_similarity(SIMILAR_TOOL_THRESHOLD)
    if len(similar_tools) > 0:
        print(f"POSSIBLE SIMILAR TOOLS ({SIMILAR_TOOL_THRESHOLD}): {len(similar_tools)}")
        for c in similar_tools:
            print(f"\t{c[0]} -> {c[1]}: {c[2]}")
    else:
        print("NO SIMILAR TOOLS")


    while True:
        print("-"*30)
        query = input("Query (q to exit): ")

        if query == 'q' or query == '':
            break

        centroid_sim_hit= tool_chooser.select_by_centroid_sim(query)
        centroid_sim_nearest= tool_chooser.select_by_centroid_sim(query=query, limit_similarity=False)
        description_sim_hit = tool_chooser.select_by_description_sim(query)

        print("CENTROID SIMILARITY HIT")
        for elem in centroid_sim_hit:
            print(f"\t {elem[1]}: {elem[0]} ")

        print("CENTROID NEAREST")
        for elem in centroid_sim_nearest:
            print(f"\t {elem[1]}: {elem[0]} ")

        print("DESCRIPTION NEAREST")
        for elem in description_sim_hit:
            print(f"\t {elem[1]}: {elem[0]} ")

        print()
        
