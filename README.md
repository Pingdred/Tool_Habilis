# TOOL-HABILIS

Tool-Habilis è vuole essere il proof of concept di un tool selection system che, in sostituzione al LLM, si occupa di selezionare il tool più adatto alla query fornita.

> **_NOTA:_**  Il progetto è ancora incompleto e potrebbe presentare numerosi bug

## Come installarlo

Clona il repo:

```bash
git clone https://github.com/sirius-0/Tool_Habilis.git
```

Entra nella cartella:

```bash
cd Tool_Habilis
```

Esegui il main passando come argomento il file JSON contenente i tool:

```bash
python main.py tools.json
```

Verrà creata una cartella `qdb_plugins` nella directory principale, contenente i file che [qdrant](https://qdrant.tech) utilizza per memorizzare le varie [collection](https://qdrant.tech/documentation/concepts/collections/) create.

> **_NOTA:_** Dopo la creazione del database, nelle successive esecuzioni si può omettere il file contenente i tool, nel caso vanga ugualmente passato come argomento i tools nel database non verrano sovrascritti.

## Come funziona

Iniziamo definendo cos'è un tool per Tool-Habilis.

Un tool è rappresentato come un oggetto JSON, ed è strutturato nel seguente modo:

```JSON
{
    "name": "nome_del_tool",
    "description": "descrizione_tool",
    "examples": [
        "esempio_1",
        "esempio_2",
        "...",
    ],
    "arguments": {
        "nome_argomento_1": "descrizione_argomento_1",
        "nome_argomento_2": "descrizione_argomento_2",
        "nome_argomento_3": "descrizione_argomento_3"
    }
}
```

Il file `tool.json` presente nel repository contiene 12 tool di esempio, ogni tool ha 5 esempi di utilizzo e può o no avere degli argomenti.

> **_NOTA:_** Gli esempi di utilizzo sono stati generati con ChatGPT per fare solo qualche rapido test di funzionamento

### Qdrant collection

Prima di spiegare il processo di creazione di un tool bisogna introdurre i due tipi di [collection](https://qdrant.tech/documentation/concepts/collections/) principali utilizzate.

Collection utilizzata per memorizzare gli esempi di utilizzo (una per ogni tool), che chiameremo `tool_examples_collection`:

```Python
client.recreate_collection(
    collection_name=tool_name,
    vectors_config=models.VectorParams(
        size= vector_size,
        distance=models.Distance.COSINE
    )
)
```

ogni punto appartenente a questa collection ha la seguente struttura:

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

Il secondo tipo di collection che chiameremo `tools_info_collection`, è utilizzata per memorizzare le informazioni relative ai tool ed ha la seguente struttura:

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

ogni punto appartenente a questa collection ha la seguente struttura:

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

## Creazione di un tool

La creazione di un tool avviene nel seguente modo:

1. Lettura del tool dal file;
2. Se non esiste la `tools_info_collection` viene creata;
3. Creazione della collection `tool_examples_collection` se esiste già un omonima la creazione del tool viene interrotta;
4. Embedding e inserimento dei vari esempi nella `tool_examples_collection` creata;
5. Calcolo del centroide (punto medio) della `tool_examples_collection`;
6. Calcolo dell'esempio meno simile al punto medio utilizzando la cosin similarity;
7. Inserimento del tool con i relativi parametri nella `tools_info_collection`

## Scelta del tool (fino ad ora)

Il tool da utilizzare al momento può essere scelto in base a due criteri

- Cosin similary con la descrizione del tool;
- Cosin similarity con il centroide;

Nel caso primo caso vengono semplicemente restituiti i primi `n` tools più simili.

Nel secondo caso invece dopo aver ottenuto i primi `n` tools più simili, vengono rimossi quelli che hanno una similarità minore di `least_similar_example` + `margin`, ovvero quelli che non ricadono nel contesto di utilizzo di quel tool.

> **_NOTA:_** Non è ancora implementato il passaggio dei parametri al tool, ma la mia idea è di far gestire il flusso della conversazione al Tool-Habilis, utilizzando il LLM come un [Extractive QA model](https://huggingface.co/tasks/question-answering), utilizzando come conteso la domanda iniziale dell'utente.

## Tools collision (fino ad ora)

Due tool che ricadono o possono ricadere nello stesso contesto di utilizzo sono in collisione.

Per ora il metodo utilizzato per calcolare la possibilita di collisione tra due tool è la similarità tra i centroidi.

L'obiettivo del controllo delle collisioni tra tool è quello di poter prevedere in modo deterministico quali tool potrebbero entrare in conflitto e gestire la situazione di conseguenza.

## Esempio di utilizzo

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
Available tools:  12
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

## Note finali

Per ora non ho effettuato alcun test per valutare questo metodo di selezione dei tool, ma indicherò qui alcuni vantaggi e svantaggi che credo questo sistema potrebbe comportare.

Vantaggi:

- Riduzione del contesto utilizzato;
- Maggiore controllo sulla scelta dei tool da usare;
- Possibilità di utilizzo dei tool anche da parte di LLM meno performanti della gpt-family (o almeno lo spero);

Svantaggi:

- È troppo difficile trovare svantaggi in qualcosa che ti appassiona;

Naturalmente ogni contributo è ben accetto.
