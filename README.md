# TOOL-HABILIS

Tool-Habilis aims to be the proof of concept of a tool selection system which, in place of the LLM, takes care of selecting the most suitable tool for the query provided.

> **_NOTE:_** The project is still incomplete and may have numerous bugs

## How to install it

Clone the repo:

```bash
git clone https://github.com/sirius-0/Tool_Habilis.git
```

Enter the folder:

```bash
cd Tool_Habilis
```

Run main passing the JSON file containing the tools as an argument:

```bash
python main.py tools.json
```

A `qdb_plugins` folder will be created in the root directory, containing the files that [qdrant](https://qdrant.tech) uses to store the various [collections](https://qdrant.tech/documentation/concepts/collections/ ) create.

> **_NOTE:_** After the creation of the database, in subsequent executions the file containing the tools can be omitted, if it is passed as an argument the tools in the database will not be overwritten.

## How does it work

Let's start by defining what a tool is for Tool-Habilis.

A tool is represented as a JSON object, and is structured as follows:

```JSON
{
     "name": "name_of_tool",
     "description": "tool_description",
     "examples": [
         "example_1",
         "example_2",
         "...",
     ],
     "arguments": {
         "topic_name_1": "topic_description_1",
         "topic_name_2": "topic_description_2",
         "topic_name_3": "topic_description_3"
     }
}
```

The `tool.json` file in the repository contains 12 example tools, some with parameters and some without,each tool has 5 usage examples.
> **_NOTE:_** The examples of use have been generated with ChatGPT to do just some quick functional tests

### Qdrant collection

Before explaining the process of creating a tool, it is necessary to introduce the two main types of [collection](https://qdrant.tech/documentation/concepts/collections/) used.

Collection used to store the usage examples (one for each tool), which we will call `tool_examples_collection`:

```Python
client.recreate_collection(
     collection_name=tool_name,
     vectors_config=models.VectorParams(
         size= vector_size,
         distance=models.Distance.COSINE
     )
)
```

each point belonging to this collection has the following structure:

```Python
client.upsert(
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
```

The second type of collection, which we will call `tools_info_collection`, is used to store information relating to the tools and has the following structure:

```Python
client.recreate_collection(
     collection_name=self.__tools_collection_name,
     vectors_config={
         "description": models.VectorParams(
             size=vector_size,
             distance=models.Distance.COSINE
         ),
         "centroid": models.VectorParams(
             size=vector_size,
             distance=models.Distance.COSINE
         )
     }
)
```

each point belonging to this collection has the following structure:

```Python
client.upsert(
     collection_name=tools_collection_name,
     points=[
         models.PointStruct(
             id=self.tools_count(),
             vector={
                 "description": embedded_tool_description,
                 "centroid": embedded_centroid
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
             }
         )
     ]
)
```

## Creation of a tool

A tool is created as follows:

1. Reading the tool from the file;
2. If it doesn't exist the `tools_info_collection` is created;
3. Creation of the `tool_examples_collection` collection, if a homonymous one already exists the creation of the tool is interrupted;
4. Embedding and inserting the various examples in the `tool_examples_collection` created;
5. Calculation of the centroid (midpoint) of the `tool_examples_collection`;
6. Calculation of the example least similar to the midpoint using cosin similarity;
7. Inserting the tool with its parameters in the `tools_info_collection`

## Tool selection (until now)

The tool to be used at the moment can be chosen on the basis of two criteria

- Cosin similary with the description of the tool;
- Cosin similarity with the centroid;

In the first case, the first most similar `n` tools are simply returned.

In the second case instead, after having obtained the first most similar `n` tools, those that have a similarity lower than `least_similar_example` + `margin` are removed, i.e. those that do not fall within the context of using that tooL.

> **_NOTE:_** Parameter passing to the tool is not yet implemented, but my idea is to let Tool-Habilis manage the conversation flow, using the LLM as an [Extractive QA model](https:/ /huggingface.co/tasks/question-answering), using the user's initial question as contested.

## Tools collision (until now)

Two tools collide if they can be used in the same context

For now the method used to calculate the possibility of collision between two tools is the similarity between the centroids.

The goal of tool collision checking is to be able to predict deterministically which tools might conflict and handle the situation accordingly.

## Usage example

```Bash
Opening tools file.
Loading tools...
         get_time: OK
         get_date: OK
         get_position: OK
         web_pilot: OK
         simple_calculator: OK
         google_search: OK
         bing_search: OK
         mysql_query: OK
         wolfram: OK
         calendar_read: OK
         calendar_create: OK
         bus: OK
Available tools: 12
POSSIBLE SIMILAR TOOLS (0.3): 3
         get_time -> get_date: 0.48529293272777696
         get_time -> get_position: 0.3427203022642258
         get_date -> calendar_read: 0.3401528591325772
------------------------------
Query (q to exit): I'm lost, where am I?
CENTROID SIMILARITY HIT
CENTROID NEAREST
          0.5300513596281098: get_position
DESCRIPTION NEAREST
          0.276624868881307: get_position

------------------------------
Query (q to exit): What time is it?
CENTROID SIMILARITY HIT
          0.84708402369244: get_time
CENTROID NEAREST
          0.84708402369244: get_time
DESCRIPTION NEAREST
          0.5366598332738072: get_time
```

## Final notes

I haven't done any tests to evaluate this method of selecting tools yet, but I will point out here some advantages and disadvantages that I believe this system could have.

Advantages:

- Reduction of the context used;
- Greater control over the choice of tools to use;
- Possibility of using the tools also by less performing LLMs than those belonging to the gpt family (or at least I hope so)

Disadvantages:

- It's too hard to find disadvantages in something you're passionate about;

Of course, any contribution is welcome.
