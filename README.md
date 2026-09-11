# Website RAG chat demo

A demo application for an AI chat website with RAG capabilities

# Known issues, limitations, and decisions

During scraping, I noticed that `https://compileit.com/kunskap/nextjs-seo` and `https://compileit.com/kunskap/strukturerad-data` link to `https://compileit.com/seo` which gives a 404 response.

The current implementation only stores chat history on the frontend. This allows a malicious user to invent their own history (for both the User and Assistant) and request completions for that.

