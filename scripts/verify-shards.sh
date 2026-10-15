#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
HOLDER=${1:?Usage: verify-shards.sh holder_index}
[[ $HOLDER =~ ^[0-2]$ ]] || exit 2
python3 - "$WEIGHTS_DIR" "$RECIPE_DIR/lists/files-holder$HOLDER.txt" <<'PY'
import hashlib
from pathlib import Path
import sys
weights, manifest = map(Path, sys.argv[1:])
checksums = {}
for line in (weights/'SHA256SUMS').read_text().splitlines():
    if line.strip():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip('*').removeprefix('./')] = checksum
names=manifest.read_text().splitlines()
for name in names:
    expected = checksums[name]
    with (weights/name).open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    if h.hexdigest()!=expected: raise SystemExit(f'FAIL: {name}')
    print(f'OK: {name}')
print(f'PASS: {len(names)} shard hashes')
PY
