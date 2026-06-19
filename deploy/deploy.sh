#!/usr/bin/env bash
set -euo pipefail

cd "$DEPLOY_PATH"

echo "==> Pulling $DEPLOY_BRANCH..."
git fetch origin "$DEPLOY_BRANCH"
git checkout "$DEPLOY_BRANCH"
git pull origin "$DEPLOY_BRANCH"

echo "==> Building and starting containers..."
NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" \
  docker compose -f "$DEPLOY_COMPOSE_FILE" up --build -d

echo "==> Pruning old images..."
docker image prune -f

echo "==> Deploy complete."
