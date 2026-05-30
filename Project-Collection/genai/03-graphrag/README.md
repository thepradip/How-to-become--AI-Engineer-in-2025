# GenAI 03 · GraphRAG 🔴

**Problem.** Flat RAG retrieves isolated chunks and struggles with "how are X and Y related?" or
multi-hop questions. **GraphRAG** builds a knowledge graph of entities + relationships, enabling
connected, multi-hop retrieval.

**What you build.** Entity extraction → a `networkx` co-occurrence graph → neighbourhood retrieval →
grounded answer (shared LLM client), in the shared chat UI.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (networkx + mock LLM)
streamlit run app.py      # "graph", then ask relationship questions
```

## What you learned
Knowledge graphs for retrieval · entities & relations · ego-graph neighbourhood retrieval · when
GraphRAG beats flat RAG. **Production:** LLM-extracted typed triples into **Neo4j**, community
summaries (Microsoft GraphRAG), hybrid graph+vector retrieval.

## Tested on
CPU (Python 3.11), fully offline (networkx + mock LLM). Real triple extraction needs an LLM.

> **Freelance relevance.** Knowledge-graph Q&A over org data is a high-value, differentiated RAG build.
