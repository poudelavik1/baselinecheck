#!/bin/bash
# Fetch the official code of the Tier 2 models into third_party/ (ignored by git).
# Run on the LOGIN node (needs internet), from the repo root:
#     bash scripts/install_third_party.sh crackformer
#     bash scripts/install_third_party.sh scsegamba        # code only; the env: scripts/install_scsegamba.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {   # fetch <folder> <url> <commit>
    if [ ! -d "third_party/$1/.git" ]; then
        git clone "$2" "third_party/$1"
    fi
    git -C "third_party/$1" fetch -q origin
    git -C "third_party/$1" checkout -q "$3"     # pinned: the commit the wrapper was written against
    echo "third_party/$1 at $(git -C "third_party/$1" rev-parse --short HEAD)"
}
for target in "$@"; do
    case "$target" in
        crackformer) fetch CrackFormer-II https://github.com/LouisNUST/CrackFormer-II.git 1d701aea7b09e85b106a74894cf4779f59fe9af6 ;;
        scsegamba)   fetch SCSegamba https://github.com/Karl1109/SCSegamba.git cc74c4606f03bc308b1ef1c07ef6229f9123c519 ;;
        *) echo "unknown target: $target (crackformer | scsegamba)"; exit 1 ;;
    esac
done
