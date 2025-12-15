#!/bin/bash
set -euo pipefail

# ---- CONFIGURATION ----
# Define your list (can be any values)
i_list=({1..10})   # expands to 1 2 3 ... 10

# ---- MAIN LOOP ----
for i in "${i_list[@]}"; do
    folder="sample_${i}"
    printf "Creating folder: %s\n" "$folder"
    mkdir -p -- "$folder"
done

