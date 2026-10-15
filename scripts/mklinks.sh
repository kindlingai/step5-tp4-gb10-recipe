#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
R=${1:?Usage: mklinks.sh rank}; rank_check "$R"
mkdir -p "$MODEL_DIR"
for holder in 0 1 2; do
  src=$STEP5_ROOT/nfs/holder$holder
  [[ $R != "$holder" ]] || src=$WEIGHTS_DIR
  while IFS= read -r f; do
    [[ -f $src/$f ]] || { echo "Missing file: $src/$f" >&2; exit 2; }
    ln -sfn -- "$src/$f" "$MODEL_DIR/$f"
  done < "$RECIPE_DIR/lists/files-holder$holder.txt"
done
src=$STEP5_ROOT/nfs/holder0; [[ $R != 0 ]] || src=$WEIGHTS_DIR
while IFS= read -r f; do
  [[ -f $src/$f ]] || { echo "Missing file: $src/$f" >&2; exit 2; }
  ln -sfn -- "$src/$f" "$MODEL_DIR/$f"
done < "$RECIPE_DIR/lists/files-small.txt"
printf 'Model view ready: %s\n' "$MODEL_DIR"
