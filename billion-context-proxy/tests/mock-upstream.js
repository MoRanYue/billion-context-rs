// Deterministic upstream used by the pass-through parity checks.
//   node tests/mock-upstream.js <port>
const http = require("node:http");

const port = Number(process.argv[2] || 18099);

const server = http.createServer((req, res) => {
  if (req.url === "/echo") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      res.writeHead(200, { "content-type": "application/json", "x-mock": "echo" });
      res.end(JSON.stringify({
        method: req.method,
        url: req.url,
        headers: req.headers,
        body,
      }));
    });
    return;
  }
  if (req.url.startsWith("/v1/messages")) {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      let received = null;
      try {
        received = JSON.parse(body);
      } catch {
        received = body;
      }
      res.writeHead(200, { "content-type": "application/json", "x-mock": "messages" });
      res.end(JSON.stringify({ received }));
    });
    return;
  }
  if (req.url === "/sink") {
    let n = 0;
    req.on("data", (c) => (n += c.length));
    req.on("end", () => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ length: n, contentLength: req.headers["content-length"] ?? null }));
    });
    return;
  }
  if (req.url === "/sse") {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-mock": "sse",
    });
    res.write("event: message_start\ndata: {\"n\":1}\n\n");
    let n = 1;
    const timer = setInterval(() => {
      n += 1;
      res.write(`event: content_block_delta\ndata: {"n":${n}}\n\n`);
      if (n >= 4) {
        clearInterval(timer);
        res.write("event: message_stop\ndata: {}\n\n");
        res.end();
      }
    }, 250);
    return;
  }
  if (req.url.startsWith("/v1/chat/completions") && !req.url.includes("/sse")) {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      let received = null;
      try {
        received = JSON.parse(body);
      } catch {
        received = body;
      }
      res.writeHead(200, { "content-type": "application/json", "x-mock": "chat" });
      res.end(JSON.stringify({
        received,
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: {
          prompt_tokens: 120,
          completion_tokens: 8,
          total_tokens: 128,
          prompt_tokens_details: { cached_tokens: 40 },
        },
      }));
    });
    return;
  }
  if (req.url === "/chat/sse" || req.url.startsWith("/v1/chat/completions/sse")) {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-mock": "chat-sse",
    });
    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}\n\n`);
    res.write(`data: ${JSON.stringify({
      choices: [],
      usage: { prompt_tokens: 300, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 100 } },
    })}\n\n`);
    res.end("data: [DONE]\n\n");
    return;
  }
  if (req.url === "/redirect") {
    res.writeHead(302, { location: "/echo" });
    res.end();
    return;
  }
  if (req.url === "/status/429") {
    res.writeHead(429, { "content-type": "text/plain", "retry-after": "7" });
    res.end("rate limited");
    return;
  }
  res.writeHead(404, { "content-type": "text/plain" });
  res.end("not found");
});

server.listen(port, "127.0.0.1", () => {
  console.log(`mock-upstream listening on ${port}`);
});
