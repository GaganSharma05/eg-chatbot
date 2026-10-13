# AWS MVP deployment runbook

## Architecture and assumptions

The MVP runs the API and Ollama as separate Docker containers on one EC2
instance. An internet-facing Application Load Balancer (ALB) terminates TLS and
forwards only to port 8000 on the instance. The ALB target-group health check is
`GET /ready`; `GET /health` remains the process-only liveness endpoint. Route 53
and ACM provide a stable HTTPS name. Vercel calls that name directly.

This topology is intentionally single-node. Conversation history is in memory,
so one Hypercorn worker is required. The catalog and vocabulary are immutable
files in the API image; a new catalog requires a new image and rolling restart.
The validated last-known-good catalog snapshot remains effective within a
running process, but the container filesystem is not treated as durable.

The repository currently configures `llama3.2:1b`, not Qwen. Start with an
8-GiB-memory x86 instance and validate latency under expected concurrency.
Bursty CPU instances can throttle sustained inference. If the intended model is
larger or CPU latency fails the acceptance target, stop and select a suitable
GPU EC2 instance or approved managed endpoint; do not silently change models.

Alternatives considered:

- ECS/Fargate is a good future home for the stateless API, but does not solve
  persistent, resource-heavy Ollama hosting and adds networking/cost for the MVP.
- App Runner has the same model split and less direct control of the local model.
- A GPU instance improves latency but is a major recurring cost driver.

## Values required from the owner

- AWS account, region, VPC and two public subnets for the ALB
- an existing DNS zone and backend hostname
- the exact Vercel production origin (no path), for `CORS_ALLOWED_ORIGINS`
- the intended Ollama model name and an accepted latency/concurrency target
- EC2 instance class after model benchmarking
- WhatsApp verify token source in Secrets Manager (the integration itself is unchanged)

## AWS resources and least privilege

Create an ECR repository, one EC2 instance with encrypted gp3 storage, an ALB,
target group, ACM certificate, Route 53 alias, CloudWatch log group, Secrets
Manager secret, and security groups. The ALB accepts 443 from the internet and
redirects 80 to 443. The instance accepts port 8000 only from the ALB security
group and administrative access only through Systems Manager Session Manager;
do not open SSH or Ollama port 11434.

The EC2 role needs only `AmazonSSMManagedInstanceCore`, read access to the one
ECR repository, read access to the one runtime secret, and log-stream write
access to `/exotic-grove/eg-chatbot`. The deployment operator separately needs
scoped ECR push and permissions to manage the listed resources. Do not attach
administrator policy to the instance.

## 1. Local validation (no AWS resources)

Copy `.env.production.example` to `.env`, replace every placeholder, and run:

```powershell
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:1b
docker compose up -d api
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
docker compose down
```

The Ollama named volume persists model data. `docker compose down -v` deletes it
and forces a model download; use that only when intentionally cleaning up.

Test a model-dependent prompt and a deterministic catalog prompt separately.
For the unavailable-model test, stop the Ollama container and confirm the
model-dependent request returns sanitized HTTP 503 while `/health` and catalog
responses remain available.

## 2. Build and publish the API image (approval and AWS credentials required)

Choose immutable values for the variables below. The commands are examples and
must not be run until the owner approves creation/use of billable AWS resources.

```powershell
$AwsRegion = "REPLACE_REGION"
$AwsAccountId = "REPLACE_ACCOUNT_ID"
$Repository = "eg-chatbot"
$ImageTag = "REPLACE_RELEASE_OR_COMMIT_ID"
$Registry = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"

aws ecr create-repository --repository-name $Repository --region $AwsRegion
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin $Registry
docker build --pull --tag "${Repository}:${ImageTag}" .
docker tag "${Repository}:${ImageTag}" "${Registry}/${Repository}:${ImageTag}"
docker push "${Registry}/${Repository}:${ImageTag}"
```

If the repository already exists, omit `create-repository`. Enable ECR scan on
push and lifecycle rules appropriate to the release-retention policy.

## 3. Provision and configure AWS (approval required)

1. Create the log group with a finite retention period.
2. Store runtime values as one Secrets Manager secret or as SecureString
   parameters. Include production paths, model settings, the exact CORS origin,
   log level, request limits, and the WhatsApp token. Never store the secret in
   user data or an AMI.
3. Launch the selected Amazon Linux 2023 EC2 instance with encrypted gp3 storage,
   the least-privilege role, SSM connectivity, Docker, and Compose v2.
4. Place this deployment Compose file at `/opt/eg-chatbot/compose.yaml`. Retrieve
   the secret through the instance role into `/opt/eg-chatbot/runtime.env`, owned
   by root with mode `0600`. Ensure the retrieval process does not print values.
5. Set `ECR_IMAGE` to the immutable image URI and `AWS_REGION` in the service
   manager environment. Log in to ECR, pull the images, and start Ollama.
6. Run `docker compose exec ollama ollama pull REPLACE_APPROVED_MODEL`. Model data
   remains on the encrypted Docker volume and is not baked into the API image.
7. Start the API and verify `/health` and `/ready` from the instance.
8. Create an ALB target group on HTTP port 8000 with `/ready` as its health path,
   then attach the instance. Use a deregistration delay of at least 30 seconds.
9. Validate the ACM certificate, configure the HTTPS listener, add HTTP-to-HTTPS
   redirect, and create the Route 53 alias.

Use a systemd unit that runs `docker compose up -d` after Docker starts. During
shutdown, run `docker compose down --timeout 30`; Hypercorn receives SIGTERM and
honors its graceful timeout.

## 4. Vercel and end-to-end verification

Set the frontend's existing backend/API environment variable to
`https://REPLACE_BACKEND_HOSTNAME`, then redeploy the existing Vercel project.
Do not move the frontend. Verify:

1. HTTPS certificate and HTTP redirect.
2. `/health` and `/ready` through the public hostname.
3. Allowed-origin preflight and chat POST from the production frontend.
4. A disallowed origin receives no CORS allow header.
5. Oversized, malformed, and schema-invalid requests return sanitized 413/400/422.
6. Product IDs and prices match the bundled catalog.
7. Model prompt latency, timeout behavior, and sanitized unavailable response.
8. CloudWatch contains operational errors but no message bodies or secrets from
   the web chat path.

## Rollback

Keep the previous immutable ECR image URI. Set `ECR_IMAGE` back to it, run
`docker compose pull api`, then `docker compose up -d --no-deps api`. Confirm the
target becomes healthy before closing the change. Catalog rollback occurs with
the image because the catalog is bundled. The Ollama model is rolled back
explicitly with `ollama pull` and the approved prior `OLLAMA_MODEL` value.

## Recovery and cleanup

If Ollama is unhealthy, restart only that container, inspect CloudWatch logs,
confirm volume capacity, and re-pull the exact approved model if corrupt. The API
returns 503 for model-dependent traffic rather than substituting another model.

To stop recurring costs after explicit approval, remove the Route 53 record,
ALB/listeners/target group, EC2 instance and volumes, unused ECR images/repository,
log group, secret, and certificate if dedicated. Retain only artifacts required
by the agreed data-retention policy.

Primary cost drivers are the always-on EC2 instance, ALB hourly/LCU charges,
gp3 storage, CloudWatch ingestion/retention, DNS, public IPv4, and data transfer.
GPU inference would dominate these costs. Obtain current regional prices with
the AWS Pricing Calculator before approval; no cost estimate in this repository
is a quote.
