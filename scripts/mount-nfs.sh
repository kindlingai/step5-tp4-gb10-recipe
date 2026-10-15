#!/usr/bin/env bash
# Run with elevated privileges on each compute node. Rank 3 owns no shards.
source "$(dirname -- "$0")/common.sh"
R=${1:?Usage: mount-nfs.sh rank}; rank_check "$R"
for holder in 0 1 2; do
  [[ $holder != "$R" ]] || continue
  target=$STEP5_ROOT/nfs/holder$holder
  mkdir -p "$target"
  if mountpoint -q "$target"; then
    echo "Already mounted: $target; verify its source and ro options with findmnt"
  else
    mount -t nfs4 -o ro,nconnect=8 "${NODES[$holder]}:$WEIGHTS_DIR" "$target"
  fi
done
