import sys

import json

from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

from tool_habilis import THabilis as th

import constants


def main():
    client = QdrantClient(path="./qdb_plugin")
    embedder = OpenAIEmbeddings(openai_api_key=constants.OPENAI_KEY)

    tool_chooser = th(client, embedder, 1536)

    if len(sys.argv) >= 2 and sys.argv[1] == '-r':
        with open('tools.json', 'r') as tools_file:
            tools = json.load(tools_file)

        print("Tools file loaded.")
        tool_chooser.load(tools['tools_list'])


    tool_chooser.print_tools_collection()

    print(tool_chooser.tools_count)

    while True:

        query = input("Query: ")

        if query == 'q' or query == '':
            break

        tool_chooser.select_by_mean(query)
        tool_chooser.select_by_description(query)
        print("-"*20)


if __name__ == "__main__":
    main()
