// Cloudflare Worker (free tier): receives the Hugging Face webhook and relays it to GitHub.
// Needed because HF webhooks can't send the Authorization header GitHub's API requires.
//
// Secrets (wrangler secret put ...): HF_WEBHOOK_SECRET, GITHUB_TOKEN
// Vars (wrangler.toml): GITHUB_REPO  e.g. "youruser/chat-mlops"

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("method not allowed", { status: 405 });

    // HF sends the secret you set on the webhook in this header
    if (request.headers.get("X-Webhook-Secret") !== env.HF_WEBHOOK_SECRET) {
      return new Response("unauthorized", { status: 401 });
    }

    const payload = await request.json();

    // Only a content update on main means a new model was published.
    // Checkpoint commits on the `checkpoints` branch are ignored.
    if (payload?.event?.scope !== "repo.content") return new Response("ignored: scope");
    const main = (payload.updatedRefs || []).find((r) => r.ref === "refs/heads/main");
    if (!main) return new Response("ignored: not main");

    const res = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "hf-webhook-relay",
      },
      body: JSON.stringify({
        event_type: "hf-model-updated",
        client_payload: { sha: main.newSha, repo: payload.repo?.name },
      }),
    });

    return new Response(`github dispatch: ${res.status}`, { status: res.ok ? 200 : 502 });
  },
};
