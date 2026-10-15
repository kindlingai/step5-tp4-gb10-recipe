#!/usr/bin/env bash
# Scan relative file names, text, and committed history. Character classes keep
# the pattern definitions from matching themselves; this file is scanned too.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
pattern='100[.]|10[.]98[.]|10[.]99[.]|d[g]x[0-9]|spa[r]k-[0-9a-f]{4}|[/]home[/]|[/]ro[o]t([/]|$)|[/]Users[/]|reg[g]ie|pu[l]se|As[t]ra|Fa[b]le|O[p]us|S[o]l|Ar[g]on|Chu[c]k|GL[M]|Qw[e]n|[.]tail[a-f0-9]+[.]ts[.]net|gh[p]_|gh[o]_|github_[p]at_|xo[x][abprs]-|AK[I]A[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
status=0
# This is a small, known repository, not a disk-wide search.
rc=0
grep -rInaE --exclude-dir=.git -- "$pattern" . || rc=$?
[[ $rc == 1 ]] || { [[ $rc == 0 ]] && status=1 || exit "$rc"; }
while IFS= read -r -d '' name; do
  if printf '%s\n' "$name" | grep -nE -- "$pattern"; then status=1; fi
done < <(find . -type d -name .git -prune -o -type f -print0)
if git rev-parse --show-toplevel >/dev/null 2>&1 && [[ $(git rev-parse --show-toplevel) == "$PWD" ]]; then
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    rc=0
    git log --all --format=fuller -p | grep -nE -- "$pattern" || rc=$?
    [[ $rc == 1 ]] || { [[ $rc == 0 ]] && status=1 || exit "$rc"; }
  fi
fi
if (( status )); then echo 'Privacy scan: FAIL'; exit 1; fi
echo 'Privacy scan: PASS (tree, relative filenames, and available commit history)'
