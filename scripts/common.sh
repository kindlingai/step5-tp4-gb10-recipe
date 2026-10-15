#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${STEP5_CONFIG:-} ]]; then
  # The configuration is a trusted shell file. Export values to the bench client.
  set -a
  source "$STEP5_CONFIG"
  set +a
fi
NODE0=${NODE0:-192.168.50.11}; NODE1=${NODE1:-192.168.50.12}
NODE2=${NODE2:-192.168.50.13}; NODE3=${NODE3:-192.168.50.14}
NODES=("$NODE0" "$NODE1" "$NODE2" "$NODE3")
STEP5_ROOT=${STEP5_ROOT:-/srv/step5}
CACHE_DIR=${CACHE_DIR:-/var/cache/step5}
WEIGHTS_DIR=$STEP5_ROOT/weights
MODEL_DIR=$STEP5_ROOT/model
NETWORK_IFACE=${NETWORK_IFACE:-eth0}
NFS_CLIENT_CIDR=${NFS_CLIENT_CIDR:-192.168.50.0/24}
SSH_USER=${SSH_USER:-}
ssh_target() { if [[ -n $SSH_USER ]]; then printf '%s@%s' "$SSH_USER" "$1"; else printf '%s' "$1"; fi; }
rank_check() { [[ ${1:-} =~ ^[0-3]$ ]] || { echo 'Expected rank 0..3' >&2; exit 2; }; }
# Consistent absolute paths preserve symlink targets inside the read-only bind mount.
[[ $STEP5_ROOT =~ ^/[a-zA-Z0-9_./-]+$ && $CACHE_DIR =~ ^/[a-zA-Z0-9_./-]+$ ]] || {
  echo 'Use absolute storage paths without spaces or shell metacharacters' >&2; exit 2;
}
