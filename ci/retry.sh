#!/usr/bin/env bash
# Re-run a command with exponential backoff to ride out transient network failures in CI
# (e.g., read timeouts while pushing the DVC cache to the WebDAV remote or fetching from GitHub).
#
# Usage: ci/retry.sh [-n MAX_ATTEMPTS] [-d BASE_DELAY] [-m MAX_DELAY] -- COMMAND [ARGS...]
#
# The command is re-run until it exits with 0 or MAX_ATTEMPTS is reached. The delay (in seconds)
# before the next attempt starts at BASE_DELAY, doubles after every failure up to MAX_DELAY, and
# gets up to 25% random jitter added. The exit code of the last attempt is returned. Only use this
# for commands that are safe to repeat (dvc pull/push, git fetch, ...).

set -u

max_attempts=3
base_delay=10
max_delay=300

usage() {
  echo "Usage: $0 [-n MAX_ATTEMPTS] [-d BASE_DELAY] [-m MAX_DELAY] -- COMMAND [ARGS...]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -n) max_attempts="$2"; shift 2 ;;
    -d) base_delay="$2"; shift 2 ;;
    -m) max_delay="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "retry.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ $# -eq 0 ]; then
  usage >&2
  exit 2
fi

attempt=1
delay=$base_delay
while true; do
  "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    if [ "$attempt" -gt 1 ]; then
      echo "retry.sh: '$*' succeeded on attempt $attempt/$max_attempts" >&2
    fi
    exit 0
  fi
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "retry.sh: '$*' failed with exit code $status after $attempt attempt(s), giving up" >&2
    exit "$status"
  fi
  jitter=$(( RANDOM % (delay / 4 + 1) ))
  sleep_for=$(( delay + jitter ))
  echo "retry.sh: '$*' failed with exit code $status (attempt $attempt/$max_attempts)," \
       "retrying in ${sleep_for}s" >&2
  sleep "$sleep_for"
  attempt=$(( attempt + 1 ))
  delay=$(( delay * 2 ))
  if [ "$delay" -gt "$max_delay" ]; then
    delay=$max_delay
  fi
done
