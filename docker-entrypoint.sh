#!/bin/sh
set -eu

exec hypercorn app.main:app \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-1}" \
    --graceful-timeout "${GRACEFUL_TIMEOUT_SECONDS:-30}"
