"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Role = "user" | "assistant";

type Source = {
  title: string;
  url: string;
};

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];
  status?: "searching" | "generating" | "stopped";
  error?: boolean;
};

type StreamEvent = {
  event: string;
  data: string;
};

class UserFacingError extends Error {}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const suggestions = [
  "Vilka tjänster erbjuder Compileit?",
  "Vilka företag har Compileit gjort appar åt?",
  "Vad skriver Compileit om AI och RAG?",
];

function createId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    buffer = buffer.replaceAll("\r\n", "\n");

    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      const rawEvent = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);

      let event = "message";
      const dataLines: string[] = [];
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) {
          event = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice("data:".length).trimStart());
        }
      }
      yield { event, data: dataLines.join("\n") };
      separator = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}

function MarkdownAnswer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
        ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-teal-700 underline decoration-teal-300 underline-offset-2 hover:text-teal-900"
          >
            {children}
          </a>
        ),
        strong: ({ children }) => <strong className="font-semibold text-slate-950">{children}</strong>,
        code: ({ children }) => (
          <code className="rounded bg-slate-200 px-1 py-0.5 font-mono text-[0.9em]">{children}</code>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const abortController = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth" });
  }, [messages, isStreaming]);

  function updateMessage(id: string, update: Partial<ChatMessage>) {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...update } : message)),
    );
  }

  async function submitQuestion(question: string) {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isStreaming) return;

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmedQuestion,
    };
    const assistantId = createId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "searching",
    };
    const requestMessages = [...messages, userMessage].map(({ role, content }) => ({ role, content }));

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setIsStreaming(true);

    const controller = new AbortController();
    abortController.current = controller;

    try {
      const response = await fetch(`${BACKEND_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: requestMessages }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new UserFacingError(`Anropet till servern misslyckades (${response.status}).`);
      }
      if (!response.body) {
        throw new UserFacingError("Servern returnerade ingen dataström.");
      }

      for await (const event of parseSSE(response.body)) {
        const data = JSON.parse(event.data) as {
          state?: "searching" | "generating";
          text?: string;
          sources?: Source[];
        };

        if (event.event === "status" && data.state) {
          updateMessage(assistantId, { status: data.state });
        } else if (event.event === "token" && data.text) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + data.text, status: undefined }
                : message,
            ),
          );
        } else if (event.event === "sources") {
          updateMessage(assistantId, { sources: data.sources ?? [] });
        } else if (event.event === "error") {
          throw new UserFacingError("Servern returnerade ett fel.");
        }
      }
    } catch (error) {
      if (controller.signal.aborted) {
        updateMessage(assistantId, { status: "stopped" });
      } else {
        updateMessage(assistantId, {
          content: error instanceof UserFacingError ? error.message : "Något gick fel. Försök igen.",
          error: true,
          status: undefined,
        });
      }
    } finally {
      abortController.current = null;
      setIsStreaming(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(input);
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(input);
    }
  }

  function stopStreaming() {
    abortController.current?.abort();
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4 sm:px-6">
      <header className="flex items-center justify-between border-b border-slate-200 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-teal-700 text-lg font-semibold text-white shadow-sm">
            C
          </div>
          <div>
            <h1 className="font-semibold tracking-tight text-slate-950">Compileits assistent</h1>
            <p className="text-sm text-slate-500">Ställ frågor om Compileits webbplats</p>
          </div>
        </div>
        <span className="hidden rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 sm:inline-flex">
          Kunskapsbas från webbplatsen
        </span>
      </header>

      <section className="flex min-h-0 flex-1 flex-col">
        <div className="flex-1 space-y-6 overflow-y-auto py-8">
          {messages.length === 0 ? (
            <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-3xl bg-teal-100 text-3xl text-teal-800">
                ✦
              </div>
              <h2 className="text-2xl font-semibold tracking-tight text-slate-950">Vad vill du veta?</h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                Fråga om Compileits tjänster, projekt, lediga tjänster eller artiklar. Svaren baseras på innehåll som har hämtats från webbplatsen.
              </p>
              <div className="mt-7 grid w-full max-w-2xl gap-3 sm:grid-cols-3">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => setInput(suggestion)}
                    className="rounded-2xl border border-slate-200 bg-white p-4 text-left text-sm leading-5 text-slate-600 shadow-sm transition hover:-translate-y-0.5 hover:border-teal-300 hover:text-slate-950"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                {message.role === "assistant" && (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-teal-100 text-sm text-teal-800">✦</div>
                )}
                <div
                  className={`max-w-[88%] rounded-3xl px-5 py-4 text-[15px] leading-7 shadow-sm ${
                    message.role === "user"
                      ? "rounded-br-md bg-slate-900 text-white"
                      : message.error
                        ? "rounded-bl-md border border-red-200 bg-red-50 text-red-800"
                        : "rounded-bl-md border border-slate-200 bg-white text-slate-700"
                  }`}
                >
                  {message.role === "assistant" && message.status && message.status !== "stopped" && !message.content ? (
                    <div className="flex items-center gap-2 text-sm text-slate-500">
                      <span className="flex gap-1">
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-500 [animation-delay:-0.2s]" />
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-500 [animation-delay:-0.1s]" />
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-teal-500" />
                      </span>
                      {message.status === "searching" ? "Söker på webbplatsen…" : "Skriver ett svar…"}
                    </div>
                  ) : message.role === "assistant" ? (
                    <>
                      <MarkdownAnswer content={message.content} />
                      {message.status === "stopped" && <p className="mt-2 text-xs text-slate-400">Genereringen avbröts.</p>}
                      {!!message.sources?.length && (
                        <div className="mt-5 border-t border-slate-200 pt-3 text-xs leading-5">
                          <p className="mb-1 font-semibold uppercase tracking-[0.14em] text-slate-400">Källor</p>
                          <ul className="space-y-1">
                            {message.sources.map((source) => (
                              <li key={source.url}>
                                <a className="text-teal-700 hover:underline" href={source.url} target="_blank" rel="noreferrer">
                                  {source.title}
                                </a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    message.content
                  )}
                </div>
              </article>
            ))
          )}
          <div ref={bottomRef} />
        </div>

        <div className="sticky bottom-0 bg-[#f7faf9] pb-5 pt-2">
          <form onSubmit={handleSubmit} className="rounded-3xl border border-slate-300 bg-white p-2 shadow-lg shadow-slate-200/60">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="Ställ en fråga om Compileit…"
              rows={2}
              disabled={isStreaming}
              className="max-h-40 min-h-12 w-full resize-none bg-transparent px-3 py-2 text-[15px] leading-6 text-slate-950 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
              aria-label="Din fråga"
            />
            <div className="flex items-center justify-between px-2 pb-1">
              <p className="text-xs text-slate-400">Enter för att skicka · Shift+Enter för ny rad</p>
              {isStreaming ? (
                <button
                  type="button"
                  onClick={stopStreaming}
                  className="rounded-xl border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-500 hover:text-slate-950"
                >
                  Avbryt
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="rounded-xl bg-teal-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
                >
                  Skicka
                </button>
              )}
            </div>
          </form>
          <p className="mt-3 text-center text-xs text-slate-400">Chatten är tillfällig och rensas när du uppdaterar sidan.</p>
        </div>
      </section>
    </main>
  );
}
