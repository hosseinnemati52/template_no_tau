#!/bin/bash

# Define your lists here
#i_list=(1 2 3 4 5 6 7 8)          # list of values for i
#j_list=(3 4 5 6 7 8)          # list of values for j

source lw_lc_lists.sh

# Optional: uncomment to see what was loaded
# echo "lw_list: ${lw_list[@]}"
# echo "lc_list: ${lc_list[@]}"

# Create folders
for lw in "${lw_list[@]}"; do
    for lc in "${lc_list[@]}"; do
        folder="lw_${lw}__lc_${lc}"
        echo "Creating folder: $folder"
        mkdir -p "$folder"

        # Copy files/folders into the new folder
        cp -R template_lw_lc/* "$folder"/
    done
done
sleep 5

# modifies l_w and l_c in params files
python3 params_modifier.py;
sleep 5

# runs terminal opener files
python3 t_o_opener.py;
