# Agentic RAG - CRAG System

this is a Corrective RAG (CRAG) system built using LangGraph and FastAPI. the idea is simple - instead of just retrieving docs and answering, the system first evaluates if the retrieved docs are actually good enough to answer the question. if not, it rewrites the query and does a web search to get better context.

---

## How it works (Architecture)

the flow goes like this:

```
User Question
     |
     v
  Retrieve          <- gets top 4 similar chunks from FAISS vector store
     |
     v
Retrieval Evaluator  <- scores each chunk 0-10, decides: correct / incorrect / ambiguous
     |
     |-- correct   --> Knowledge Refinement
     |-- incorrect --> Rewrite Query --> Web Search --> Knowledge Refinement
     |-- ambiguous --> Rewrite Query --> Web Search --> Knowledge Refinement
                                                            |
                                                            v
                                                        Generate   <- answers from refined context
                                                            |
                                                    answergiven? --> end
                                                            |
                                                         retry  --> back to Retrieve (max 3 times)
```

**the 3 evaluation states:**
- `correct` - at least one chunk scored >= 8, use the docs directly
- `incorrect` - all chunks scored < 4, skip docs and go straight to web search
- `ambiguous` - somewhere in between, use both docs + web search

**Knowledge Refinement** - after getting the context (from docs or web), it breaks everything into sentences and filters out only the sentences that are actually relevant to the question. this way the generator gets clean focused context instead of a big blob of text.

---

## Folder Structure

```
Assinment/
├── app.py                  # FastAPI server - all the endpoints live here
├── rag/
│   ├── state.py            # App_State TypedDict - the shared state across all nodes
│   ├── loader.py           # loads PDFs, splits into chunks, builds FAISS vector store
│   ├── nodes.py            # all the langgraph node functions
│   └── graph.py            # wires all the nodes together and compiles the workflow
├── Models/
│   └── OutputSchema.py     # pydantic schemas for structured LLM outputs
├── Utils/
│   └── HelperFxns.py       # helper - text to sentence converter
├── Documents/              # put your PDFs here
├── Doc_Rag.ipynb           # original notebook (for reference)
├── .env.example
└── requirement.txt
```

---

## Setup

**1. clone the repo and create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # windows
source venv/bin/activate     # mac/linux
```

**2. install dependencies**
```bash
pip install -r requirement.txt
```

**3. setup your API keys**

copy `.env.example` to `.env` and fill in your keys
```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

- OpenAI key -> https://platform.openai.com/api-keys
- Tavily key -> https://app.tavily.com (free tier available)

**4. add your PDFs**

drop any PDF files into the `Documents/` folder. the system will load them automatically on startup.

---

## How to Run

```bash
uvicorn app:app --reload
```

server will start at `http://localhost:8000`

you can also check the auto generated API docs at `http://localhost:8000/docs`

---

## API Endpoints

### GET /health
just checks if the server is running


---

## Design Decisions and Tradeoffs

**why LangGraph ?**
i needed a way to have conditional routing in the pipeline - like if docs are good go here, if not go there. LangGraph makes this really clean with its node/edge model. doing this manually would have been a mess.

**why FAISS over a cloud vector DB ?**
for this project FAISS is fine - it runs locally, no extra setup, no cost. the tradeoff is it loads everything into memory and doesnt persist between restarts. for a real prod system you'd want something like Pinecone or Weaviate.

**the 3 level evaluation (correct/incorrect/ambiguous)**
instead of a simple yes/no on the retrieved docs, i went with 3 levels. this way ambiguous cases can still use the local docs but also supplement with web search. a simple yes/no would either waste the good parts of the docs or miss important web context.

**knowledge refinement step**
after getting context (docs or web), i break it into sentences and filter only the relevant ones. this keeps the context clean and focused. the tradeoff is it makes extra LLM calls per sentence which adds latency and cost. but the answer quality is much better.

**vector store loads on startup**
the FAISS vector store is built once when the server starts. this means if you upload a new PDF via `/upload-pdf`, the new doc wont be searchable until you restart the server. a better approach would be to rebuild the index after each upload but that adds complexity.

**retry goes back to Retrieve**
when the generator cant answer, the retry loops back to the Retrieve node with the same question. ideally it should rewrite the query first before retrying but thats a future improvement.
