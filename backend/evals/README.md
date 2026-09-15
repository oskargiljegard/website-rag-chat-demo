# Local API evaluations

This evaluator calls the running chat API rather than importing the graph. That
keeps the evaluation stable if the graph, retriever, or storage implementation
changes later.

## Run the backend

From `backend/`:

```powershell
uv run fastapi dev
```

## Run an evaluation

From `backend/`:

```powershell
uv run evals/run_eval.py
```

Results are written to `evals/results/` as one pretty-printed JSON array per run
and one summary JSON file per run. The evaluator always runs the complete dataset and
records the final answer, page-level sources, optional evidence excerpts, usage,
cost, and streaming timings. Each dataset question is run `NUM_RUNS` times;
`NUM_RUNS = 5` in `run_eval.py`, so the result file contains 5 records per question.

Answer correctness is judged with an additional LLM call. Its cost is reported
separately from the application query cost.

## Cost calculation

The API always reports token usage. Costs are calculated using the model and
pricing constants in `rag.py`, so the evaluator and API use the same rates.
Update those constants if the models change.

Optional environment variables are:

```text
EVAL_API_URL=http://127.0.0.1:8000
EVAL_TIMEOUT_SECONDS=180
```

The agent graph is selected by default. Set `RAG_GRAPH=simple` before starting the
backend to use the original fixed retrieve-then-answer graph instead.
