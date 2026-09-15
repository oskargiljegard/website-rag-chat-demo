# Website RAG chat demo

A demo application for an AI chat website with RAG capabilities

# Prerequisites

- Create `backend/.env` with your `OPENAI_API_KEY`
- Install Node.js/npm
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

# Setup

# Creating the data

Run the preprocessing scripts in order:

```powershell
cd backend
uv run scraping/crawl.py
uv run scraping/chunk_pages.py
uv run scraping/index_chunks.py
```

# Running the system

Start the backend API:

```powershell
cd backend
uv run fastapi dev
```

Start the frontend development server:

```powershell
cd frontend
npm install
npm run dev
```

Run the evaluation suite while the backend is running:

```powershell
cd backend
uv run evals/run_eval.py
```

The evaluation runs every dataset question five times. The default backend graph is
agentic: it can call `fetch_documents` repeatedly, with five documents per call.
Set `RAG_GRAPH=simple` before starting the backend to use the original fixed
retrieve-then-answer graph.

# Design decisions

A big part of this project was the Eval to be able to measure the performance of the system. The preprocessing and the frontend had lower priority, in order to prioritise getting measurably good answers from the system.

## Preprocessing

An initial script crawls the website and stores it locally. It intentionally ignores subdomains (except www), for example https://careers.compileit.com/ since that seemed to use a different tech stack which may make the next pre-processing steps more difficult. A raw dump of all html pages is a great source of truth, and allows agents like Codex to quickly find answers in there instead of having to go to the actual website.

A second script converts the html to chunks based on html tags. The chunks are both stored as chunks.jsonl for the next script, and as more easily viewable markdown files.

A final script takes those chunks, creates embeddings and stores them in chromaDB.

Keeping these as three separate steps with persistent storage inbetween makes it easier to iterate on one without having to rerun everything.

## LangGraph and the graphs

Langgraph was chosen because I initially wanted to try out many different architectures for the assistants. In the end only two were tried but the design of the system makes it easy to try new ones.

The baseline was a simple graph that just retrieved 5 documents first and answered with an LLM second. It is very important to build a simple baseline first to know whether a more complex agent is actually better.

The second graph is an agent which uses fetch_documents as a tool.

## Frontend

The frontend is a very simple NextJS application. It sends requests to an endpoint defined with FastAPI which streams the result as SSE back. It is important for an AI assistant to provide sources for its claims, so the frontend also provides links to the pages that contain the chunks that were used when generating the answer.

## Eval

One of the most important parts of an AI assistant is an Eval, since that provides a way to measure performance.

I took 5 questions, let Codex use the raw HTML files to provide good answer criterias and expected sources, and used that to create a dataset. The eval checks those questions against the FastAPI endpoint. It measures cost, latency, how well we found the expected sources, and how well the question was answered.

The eval results show that the simple graph performs substantially better than the agent graph across answer quality, source correctness, cost, and time to first token.

| Graph | Answer score | Source score | Application cost | Median time to first token |
| --- | ---: | ---: | ---: | ---: |
| Agent graph | 0.5128 | 72% | $0.02719848 | 5.42 s |
| Simple graph | 0.6240 | 80% | $0.01267600 | 1.21 s |

Disclaimers:
- Having 5 questions is way too low for a good eval
- The answer criteria are not perfect. E.g. including "hello@compileit.com" is currently impossible because the chunking strategy removes it

# Known issues, limitations

During scraping, I noticed that `https://compileit.com/kunskap/nextjs-seo` and `https://compileit.com/kunskap/strukturerad-data` link to `https://compileit.com/seo` which gives a 404 response.

The current implementation only stores chat history on the frontend. This allows a malicious user to invent their own history (for both the User and Assistant) and request completions for that.

The current implementation sends additional SSEs that the eval need in order to track usage. This is unnecessary for the frontend and should maybe be designed as two separate endpoints.

# Possible future work

Define a middle ground between the raw HTML pages and the finished chunks, where the entire HTML pages are stored as OKF-files and viewable as an obsidian vault. This would make it easy to explore the graph of pages and inspect the data the AI will have access to.

There is a chat application available at https://compileit.com/chat. It would be interesting to compare the answers it gives to the answers this system gives.
