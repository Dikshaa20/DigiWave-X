# DigiWave X

AI-verified virtual laboratory for Digital Communication practicals.

## Status

Phase 1 complete: ground-truth oracle (NRZ-L, Manchester, AMI) with tests.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the tests

```bash
cd backend
pytest
```

All 24 tests should pass. These verify the encoding rules against
hand-worked examples, so they are the safety net for every component
built on top of the oracle.

## Try the oracle

```bash
cd backend
python -c "from app.oracle.encoders import encode; print(encode('ami',[1,0,1,1,0]))"
```

## Project layout

```
backend/
  app/
    oracle/      encoders.py  <- Phase 1 (done)
    ingestion/   Phase 2: chunking + ChromaDB
    graph/       Phase 4: LangGraph nodes
    llm/         Phase 3: provider-agnostic LLM interface
  tests/
  scripts/
frontend/        Phase 5: Streamlit pages
corpus/          your lab manual / textbook PDFs go here
data/            SQLite eval log lives here
```

## Build order

1. **Oracle** — done
2. Ingestion + retrieval (ChromaDB, test retrieval before adding any LLM)
3. Generator (first LLM call, needs an API key)
4. Verifier + Confidence Scorer, then wire the full LangGraph
5. Streamlit frontend
6. Diagnosis Agent, Hint Escalation, What-If, Student Profile
