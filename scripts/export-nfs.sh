#!/usr/bin/env bash
# Run with elevated privileges on storage holders 0, 1, and 2.
source "$(dirname -- "$0")/common.sh"
[[ -d $WEIGHTS_DIR ]] || { echo 'Weights directory missing' >&2; exit 2; }
# Requires a running kernel NFS server. No custom server image is required.
exportfs -i -o ro,sync,no_subtree_check "$NFS_CLIENT_CIDR:$WEIGHTS_DIR"
exportfs -v
