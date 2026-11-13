# ₹0 Hybrid MLOps: Colab T4 → Hugging Face → local Kubernetes → public API

```
Colab T4 (train_colab.ipynb)  ──push if val loss improves──►  HF Hub (main)
   HF webhook ► Cloudflare Worker (worker/) ► GitHub Actions (self-hosted runner)
   ► kubectl set env MODEL_REVISION=<sha> ► rolling update of chat-api pod
   ► Cloudflare Tunnel ► https://your-api ► your website fetch()
```

Layout: `server/` (FastAPI + Dockerfile), `k8s/` (manifests), `worker/` (webhook relay), `.github/workflows/deploy.yml`.
Put `train_colab.ipynb` in the repo root too.

Replace the placeholders: `your-hf-username`, `your-github-username`, `your-website.example`.

## 1. Build and run the cluster (one time)

```bash
# cluster (pick one)
minikube start --memory=8g --cpus=4        # or: kind create cluster

# build the image and load it into the cluster
docker build -t chat-api:local server/
minikube image load chat-api:local         # kind: kind load docker-image chat-api:local

kubectl apply -f k8s/chat-api.yaml
kubectl -n chat create secret generic chat-secrets \
  --from-literal=HF_TOKEN=hf_xxx \
  --from-literal=API_KEY=$(openssl rand -hex 16)
kubectl -n chat rollout status deploy/chat-api --timeout=15m   # needs a model on the Hub: run the notebook first
```

The HF token only needs **read** access. Test locally:

```bash
kubectl -n chat port-forward svc/chat-api 8000:8000
curl localhost:8000/version
curl -X POST localhost:8000/chat -H 'content-type: application/json' -H 'x-api-key: <API_KEY>' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## 2. Public URL

- **Quick test** (no account): `kubectl apply -f k8s/cloudflared-quick.yaml`, then
  `kubectl -n chat logs deploy/cloudflared-quick | grep trycloudflare`. The URL changes on restart.
- **Permanent** (free Cloudflare account + a domain on Cloudflare): create a tunnel in Zero Trust → Networks → Tunnels,
  add a public hostname pointing to `http://chat-api.chat.svc.cluster.local:8000`, then follow the comments in `k8s/cloudflared.yaml`.

## 3. Auto-deploy on new model

1. **Self-hosted runner** on your machine: GitHub repo → Settings → Actions → Runners → New self-hosted runner, and follow the commands.
   `kubectl` on that machine must point at your cluster.
2. **Relay worker**:
   ```bash
   cd worker && npx wrangler deploy
   npx wrangler secret put HF_WEBHOOK_SECRET     # any random string
   npx wrangler secret put GITHUB_TOKEN          # fine-grained PAT: this repo, "Contents: read & write"
   ```
3. **HF webhook**: huggingface.co/settings/webhooks → target URL = your worker URL, secret = the same string,
   watch your model repo (Repo updates).

Now each time the notebook pushes to `main`, the pod is rolled to that exact commit SHA. If the new pod never turns Ready,
the workflow runs `rollout undo`, which restores the previous SHA, so the old model keeps serving. Manual deploy or rollback:
Actions → Deploy model → Run workflow with a commit SHA or tag.

## 4. Call it from your website

```js
const r = await fetch("https://YOUR-TUNNEL-URL/chat", {
  method: "POST",
  headers: {"content-type": "application/json", "x-api-key": "<API_KEY>"},
  body: JSON.stringify({messages: [{role: "user", content: "Hi!"}]}),
});
const {reply} = await r.json();
```

An API key in browser JavaScript is visible to anyone, so treat it as a speed bump. For real protection, set `ALLOWED_ORIGINS`
to your site and add Cloudflare rate-limiting rules, or put the call behind a small backend.

## Notes
- Generation is CPU-only and serialized: expect a few seconds per reply from a 0.5B model, and requests queue up under load.
- The `checkpoints` branch is ignored by the relay, so only finished, improved models trigger a deploy.
- Your PC must be on for the API to be up. That is the trade-off of ₹0 local hosting.
