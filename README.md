# Website RAG chat demo

A demo application for an AI chat website with RAG capabilities

# Prerequisites

- Create `backend/.env` with your `OPENAI_API_KEY`
- Install Node.js/npm
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

# Setup

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

# Known issues, limitations, and decisions

During scraping, I noticed that `https://compileit.com/kunskap/nextjs-seo` and `https://compileit.com/kunskap/strukturerad-data` link to `https://compileit.com/seo` which gives a 404 response.

The current implementation only stores chat history on the frontend. This allows a malicious user to invent their own history (for both the User and Assistant) and request completions for that.

# Possible future work

Use the eval to compare different RAG architectures, such as an agent that continuously fetches more chunks until it believes it has enough.

Define a middle ground between the raw HTML pages and the finished chunks, where the entire HTML pages are stored as OKF-files and viewable as an obsidian vault. This would make it easy to explore the graph of pages and inspect the data the AI will have access to.