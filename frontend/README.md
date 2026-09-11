# Compileit chat frontend

This is the Next.js frontend for the Compileit RAG chat demo.

## Run locally

Start the backend from `backend/`:

```bash
uv run fastapi dev
```

Then start the frontend from `frontend/`:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend sends chat
requests to `http://localhost:8000` by default. To use another backend URL,
create `frontend/.env.local` with:

```text
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

The chat is intentionally single-session and in-memory: refreshing the page
clears the conversation.

## Useful commands

```bash
npm run lint
npm run build
```
