# Exotic Grove Produce Chatbot

A standalone Quart/ASGI API for the guardrailed Exotic Grove web chatbot.

The currently validated runtime is Python 3.13. Direct dependencies are pinned
in `requirements.txt` to make local and deployment builds reproducible.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and adjust values if necessary.
4. Start the API:

   ```powershell
   hypercorn app.main:app --bind 127.0.0.1:5000
   ```

Catalog paths are resolved relative to the repository root, not the shell's
current working directory. In production, `CATALOG_PATH`, `OLLAMA_BASE_URL`,
and `OLLAMA_MODEL` must be set explicitly.

`MAX_REQUEST_BODY_BYTES` limits HTTP request bodies. `MAX_MESSAGE_LENGTH`
separately limits accepted chatbot messages.

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Expected response:

```json
{"status":"ok"}
```

This is a liveness check and intentionally does not depend on the catalog or
LLM service.

## Readiness check

```powershell
Invoke-RestMethod http://127.0.0.1:5000/ready
```

A valid catalog snapshot returns HTTP 200 with `{"status":"ready"}`. If the
catalog cannot be initialized, readiness returns HTTP 503 with
`{"status":"not_ready"}`. Startup also fails when the initial catalog cannot
be validated. Health responses do not expose catalog paths or diagnostics.

## Chat API

`POST /api/v1/chat` accepts a JSON object containing `message` and an optional
`session_id`. Responses contain `answer`, `in_scope`, `intent`,
`product_domain`, and `identified_products`.

Conversation history and product-interest records are currently held only in
the application process. They are lost on restart and are not shared between
multiple workers; a durable shared implementation is required before relying
on this data in production.

## Product resolution and routing

Product recognition has one authoritative integration point. `DomainRouter`
invokes its injected `ProductResolver` once per accepted message and retains
the structured resolution on `DomainRoute`. The router owns generic brand,
domain, and informational-intent decisions; it does not contain product names
or aliases. `CatalogRequestDetector` separately decides whether the message
requires catalog data by combining generic commerce language with the
structured resolution.

Stable concepts and aliases live in `app/services/product_vocabulary.json`.
Each concept may explicitly list authoritative catalog product IDs. Dynamic
catalog matches carry the actual catalog `id`. `ChatOrchestrator` uses only
these explicit IDs for catalog lookup—it never assumes that a canonical
concept ID is a catalog ID.

Ambiguous aliases return a deterministic clarification and never trigger a
catalog lookup. A recognized concept with no current catalog mapping is
treated as unavailable only for explicit catalog requests; informational
questions continue through the existing LLM and output-guardrail path. An
explicit unknown request retains its original requested phrase for the safe
missing-product response.

Intent handling applies explicit precedence rules. Programming requests remain
out of scope even when they mention products or commerce terms. Pure company
questions use the deterministic brand response, while brand-qualified product
and category questions use catalog data. Negated purchase wording does not by
itself request catalog data, although an explicit price, availability, stock,
or delivery question still does.

Multi-product catalog requests resolve every mentioned item. Available items,
missing mappings, and ambiguous aliases are all represented in one
deterministic response; duplicate outcomes are collapsed without selecting an
ambiguous candidate. This path does not depend on the LLM to preserve product
completeness.

The resolver derives catalog vocabulary from the current valid snapshot on
each request. Successful refreshes therefore add, rename, or remove dynamic
matches immediately. Failed refreshes retain the previous valid snapshot, and
no separate resolver cache or refresh process exists.

## AWS deployment

Container artifacts and the controlled MVP deployment procedure are documented
in [`deploy/aws/README.md`](deploy/aws/README.md). They prepare the deployment
but do not create AWS resources.
