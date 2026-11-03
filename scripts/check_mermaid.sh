#!/usr/bin/env bash
# Render every ```mermaid block in docs/方法图解.md with mmdc; non-zero exit on any parse error.
set -euo pipefail
doc="${1:-docs/方法图解.md}"
tmp="$(mktemp -d)"
awk -v d="$tmp" '/^```mermaid/{n++; f=sprintf("%s/b%02d.mmd", d, n); on=1; next} /^```/{on=0} on{print > f}' "$doc"
for f in "$tmp"/*.mmd; do mmdc -q -i "$f" -o "${f%.mmd}.svg" >/dev/null; done
echo "ok: $(ls "$tmp"/*.mmd | wc -l) mermaid blocks rendered"
