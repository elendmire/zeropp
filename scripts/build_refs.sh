#!/usr/bin/env bash
# Build refs.bib from the curated DOI list by fetching canonical BibTeX
# from doi.org content negotiation. Never writes an entry from memory.
#
# Usage:
#   ./scripts/build_refs.sh                      # all blocks
#   ./scripts/build_refs.sh B8 B8b               # selected blocks only
#
# Output: docs/references/refs.bib  (rebuilt from scratch each run)
#         docs/references/failed_dois.txt  (anything doi.org would not resolve)

set -uo pipefail

TSV="docs/references/selected_dois.tsv"
OUT="docs/references/refs.bib"
FAILED="docs/references/failed_dois.txt"

[ -f "$TSV" ] || { echo "missing $TSV" >&2; exit 1; }

: > "$OUT"
: > "$FAILED"

# Captured before the read loop below: the function is always called as
# `wanted_block "$block" "$@"`, so inside the function $# is at least 1 even
# when the script itself received zero filter args — checking $# inside the
# function can never detect "no filter" correctly. n_filter_args records the
# script's own original argument count instead.
n_filter_args=$#

wanted_block() {
  local b="$1"; shift
  [ "$n_filter_args" -eq 0 ] && return 0
  for w in "$@"; do [ "$b" = "$w" ] && return 0; done
  return 1
}

n_ok=0
n_fail=0

# skip header
tail -n +2 "$TSV" | while IFS=$'\t' read -r block bibkey doi label priority; do
  [ -z "${doi:-}" ] && continue
  wanted_block "$block" "$@" || continue

  response=$(curl -sL --max-time 30 -w '\nHTTP_STATUS:%{http_code}' \
        -H "Accept: application/x-bibtex; charset=utf-8" \
        "https://doi.org/${doi}")
  http_code=$(printf '%s' "$response" | grep -o 'HTTP_STATUS:[0-9]*$' | cut -d: -f2)
  bib=$(printf '%s' "$response" | sed '$ s/HTTP_STATUS:[0-9]*$//')

  # A DOI that fails to resolve on doi.org's end often still returns HTTP 200
  # with an HTML "DOI Not Found" error page rather than an HTTP error code --
  # and that page can contain a line starting with "@" (e.g. a CSS @media
  # rule), which used to fool a naive "does any line start with @" check into
  # silently accepting the HTML as a valid bib entry. Verified on this exact
  # dataset: this is what happened to a truncated legacy-AMS DOI. Guard against
  # it explicitly: reject non-200, reject any HTML markers anywhere in the
  # body, and require the first non-blank character of the body to be "@".
  is_valid=1
  [ "$http_code" = "200" ] || is_valid=0
  [ -n "$bib" ] || is_valid=0
  if [ "$is_valid" -eq 1 ] && printf '%s' "$bib" | grep -qi '<!doctype\|<html'; then
    is_valid=0
  fi
  if [ "$is_valid" -eq 1 ]; then
    first_line=$(printf '%s' "$bib" | grep -m1 '[^[:space:]]' | sed 's/^[[:space:]]*//')
    case "$first_line" in
      @*) ;;
      *) is_valid=0 ;;
    esac
  fi

  if [ "$is_valid" -ne 1 ]; then
    echo "FAILED  $block  $bibkey  $doi  -- $label (http=${http_code:-?})" | tee -a "$FAILED" >&2
    n_fail=$((n_fail+1))
    sleep 1
    continue
  fi

  # Replace the publisher-assigned key with our normalised bibkey so that
  # \cite{} keys stay stable and human-readable across rebuilds.
  # (Uses awk, not sed: GNU sed's `0,/regex/` range address and empty-pattern
  # `s//repl/` reuse are not portable to BSD/macOS sed, which silently no-ops
  # instead of erroring — this was verified to break the substitution for
  # every single entry on macOS.)
  normalised=$(printf '%s' "$bib" | awk -v newkey="$bibkey" '
    NR==1 {
      match($0, /@[A-Za-z]+\{/)
      if (RSTART > 0) {
        prefix = substr($0, 1, RSTART+RLENGTH-1)
        rest = substr($0, RSTART+RLENGTH)
        sub(/^[^,]*,/, newkey ",", rest)
        print prefix rest
        next
      }
    }
    { print }
  ')

  {
    printf '%% [%s] %s\n' "$block" "$label"
    printf '%s\n\n' "$normalised"
  } >> "$OUT"

  echo "ok      $block  $bibkey"
  n_ok=$((n_ok+1))
  sleep 1   # be polite to doi.org
done

echo
echo "wrote $OUT"
grep -cE '^\s*@' "$OUT" 2>/dev/null | xargs -I{} echo "entries: {}"
if [ -s "$FAILED" ]; then
  echo "FAILURES recorded in $FAILED -- resolve these manually, do not invent them"
fi
