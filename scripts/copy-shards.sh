#!/usr/bin/env bash
# Run on the machine with the complete pinned download. No GPU work.
source "$(dirname -- "$0")/common.sh"
SOURCE_DIR=${SOURCE_DIR:-/mnt/model-source}
[[ -f $SOURCE_DIR/SHA256SUMS ]] || { echo 'Missing source SHA256SUMS' >&2; exit 2; }
for holder in 0 1 2; do
  target=$(ssh_target "${NODES[$holder]}")
  ssh "$target" "mkdir -p '$WEIGHTS_DIR'"
  rsync -aS --files-from="$RECIPE_DIR/lists/files-holder$holder.txt" "$SOURCE_DIR/" "$target:$WEIGHTS_DIR/"
  # All holders need checksums for local verification. Other metadata lives on holder 0.
  rsync -a "$SOURCE_DIR/SHA256SUMS" "$target:$WEIGHTS_DIR/SHA256SUMS"
done
rsync -a --files-from="$RECIPE_DIR/lists/files-small.txt" "$SOURCE_DIR/" "$(ssh_target "$NODE0"):$WEIGHTS_DIR/"
echo 'Copy complete. Verify each holder locally before mounting or launching.'
