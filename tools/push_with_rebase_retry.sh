#!/usr/bin/env bash
set -euo pipefail

branch="${1:?Usage: push_with_rebase_retry.sh <branch>}"
max_attempts="${PUSH_RETRY_ATTEMPTS:-5}"

if [[ ! "$branch" =~ ^[0-9A-Za-z._/-]+$ ]] || [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid branch or PUSH_RETRY_ATTEMPTS value." >&2
  exit 2
fi

for ((attempt = 1; attempt <= max_attempts; ++attempt)); do
  if ! git pull --rebase origin "$branch"; then
    git rebase --abort 2>/dev/null || true
    echo "Could not rebase generated artifacts onto origin/$branch." >&2
    exit 1
  fi

  if git push origin "HEAD:$branch"; then
    exit 0
  fi

  if ((attempt < max_attempts)); then
    echo "Push raced with another artifact writer; retrying ($attempt/$max_attempts)." >&2
    sleep $((attempt * 2))
  fi
done

echo "Could not push generated artifacts after $max_attempts attempts." >&2
exit 1
