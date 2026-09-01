# Deploying Zynaptrix Copilot

One VM, two containers (`backend` + `frontend`), everything else is external SaaS
(Neon Postgres, Cloudinary, Alibaba DashScope, optional InfluxDB Cloud).

```
browser ──HTTP :80──> frontend (Next.js)
browser ──HTTP :8000─> backend  (FastAPI)  ──> Neon / Cloudinary / DashScope
```

---

## 0. Before you touch a cloud console

**The backend image is large.** It bundles TensorFlow + PyTorch + YOLO + Mobile-SAM
+ EasyOCR: roughly **6–7 GB on disk** and it needs **~3–4 GB RAM** just to import and
serve. Plan for an instance with **≥ 4 GB RAM (8 GB comfortable) and ≥ 40 GB disk**.
A 1 vCPU / 1 GB box will not run it.

If you only need the live demo (anomaly detection + querying manuals that are already
ingested into Neon), you can later split `backend/requirements.txt` into a lean
"serve" set — drop `torch`, `torchvision`, `ultralytics`, `transformers`,
`sentence-transformers`, `easyocr`, `camelot-py`, `streamlit` (all ingestion-only) and
keep `tensorflow`. That roughly halves the image. Not required to get started.

---

## 1. Provision the server (Alibaba Cloud ECS)

New Alibaba Cloud accounts get a free ECS trial (individual: 1 vCPU/1 GB for 12 months
**or** 2 vCPU/2 GB for 3 months) **plus** a pool of pay-as-you-go trial credits
(~US$300+, valid ~60 days). The free instance is too small for this backend, so use the
**trial credits** on a bigger pay-as-you-go instance for the competition window, then
release it. Check current terms: https://www.alibabacloud.com/campaign/free-trial

1. **Region:** pick **Singapore**. DashScope's free Qwen quota is the Singapore
   (`dashscope-intl`) endpoint, and co-locating keeps latency low.
2. **Instance:** ~2 vCPU / 8 GB burstable (e.g. `ecs.e` / `ecs.t6` family),
   pay-as-you-go.
3. **Image:** Ubuntu 22.04 LTS, 40 GB+ system disk.
4. **Public IP:** assign one (or an EIP).
5. **Security Group — allow inbound:**
   | Port | Source | Why |
   |---|---|---|
   | 22 | your IP only | SSH |
   | 80 | 0.0.0.0/0 | frontend |
   | 8000 | 0.0.0.0/0 | backend API + WebSocket |
   | 443 | 0.0.0.0/0 | only if you add TLS + a domain |

## 2. Install Docker

```bash
ssh root@YOUR_SERVER_IP

curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# 2 GB swap — cheap insurance against OOM during image build / model load
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 3. Get the code and configure

```bash
git clone https://github.com/imanshadilshan/AI-Buildathon-Zynaptrix-gen-ai-copilot.git
cd AI-Buildathon-Zynaptrix-gen-ai-copilot/deploy

cp .env.example .env
nano .env
```

Set at minimum:
- `NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP:8000`
- `FRONTEND_URL=http://YOUR_SERVER_IP`
- `AI_API_KEY=...` (DashScope)
- `DATABASE_URL=...` (Neon, with `?sslmode=require`)
- `CLOUDINARY_*`

InfluxDB can stay blank for a demo.

## 4. Build and run

```bash
docker compose up -d --build      # first build ~15–25 min (heavy ML wheels)
docker compose logs -f backend    # wait for "Application startup complete"
```

Check:
- `curl http://localhost:8000/health` → `{"status":"healthy",...}`
- open `http://YOUR_SERVER_IP/` in a browser

## 5. Updating

```bash
git pull
docker compose up -d --build
```

---

## Adding a domain + HTTPS (optional, recommended if judges use it)

Browsers block `ws://` and `http://` calls from an `https://` page, so if you want
HTTPS you need it on **both** services. Easiest path:

1. Point `app.example.com` and `api.example.com` (A records) at the server IP.
2. Add a `caddy` service (image `caddy:2-alpine`, ports 80/443) with a `Caddyfile`:
   ```
   app.example.com { reverse_proxy frontend:3000 }
   api.example.com { reverse_proxy backend:8000 }
   ```
   Caddy fetches Let's Encrypt certs automatically.
3. In `.env`: `NEXT_PUBLIC_API_URL=https://api.example.com`,
   `FRONTEND_URL=https://app.example.com`; drop the public `ports:` on
   `frontend`/`backend` and expose them only to Caddy. Rebuild.

---

## Why here and not Azure?

Same shape on Azure (a VM + Docker, or Azure Container Apps). The reason to stay on
Alibaba Cloud: the Qwen models run on Alibaba's DashScope, the **free Qwen quota is
Singapore-region**, and same-cloud same-region calls are faster and don't leave the
provider network. Put the compute next to the model.
