# Deploy the live app to Google Cloud Run (free tier)

Cloud Run scales to zero when idle, so a demo app stays within the always-free
allowance (2M requests/mo, 360k GiB-s, 180k vCPU-s). Billing must be enabled on
the account, but this workload does not exceed the free tier.

## One-time setup

1. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
2. ```bash
   gcloud auth login
   gcloud projects create jewellery-match-demo            # or reuse an existing project
   gcloud config set project jewellery-match-demo
   ```
3. Link a billing account (Console → Billing) and enable the APIs:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
4. The image build downloads Torch + Fashion-CLIP + U^2-Net and builds the feature
   index, so give Cloud Build more time:
   ```bash
   gcloud config set builds/timeout 1800s
   ```

## Deploy

From the repo root:

```bash
gcloud run deploy jewellery-match \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 4 \
  --timeout 300 \
  --min-instances 0
```

`gcloud` builds the `Dockerfile`, pushes the image, and returns a public HTTPS URL
like `https://jewellery-match-xxxxxxxx-el.a.run.app`. First request after idle
cold-starts in ~15-30s (model load); subsequent requests are fast.

## Redeploy after code changes

Same `gcloud run deploy --source .` command. To tear down: `gcloud run services delete jewellery-match --region asia-south1`.

---

### Alternatives

* **Fly.io** — `fly launch` (uses the same Dockerfile), set VM to 1024 MB,
  `fly deploy`. Free allowance covers a scale-to-zero demo; card required.
* **Render / Railway** — point at this repo, Docker environment, 1 GB instance
  (paid, ~$7/mo) — simplest but not free.
