#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:52:44 2025

@author: hossein
"""


import json
import numpy as np

# lw_list = np.linspace(1,8,8)
# lc_list = np.linspace(3,8,6)

lw_list = []
lc_list = []

with open("lw_lc_lists.sh", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("lw_list"):
            values = line.split("=", 1)[1].strip("() ;")
            lw_list = [x for x in values.split()]
        elif line.startswith("lc_list"):
            values = line.split("=", 1)[1].strip("() ;")
            lc_list = [x for x in values.split()]

for lw_ind in range(len(lw_list)):
    lw = float(lw_list[lw_ind])
    
    for lc_ind in range(len(lc_list)):
        lc = float(lc_list[lc_ind])
        
        
        
        folder_name = "lw_"+str(lw)+"__lc_"+str(lc)+"/src/"

        # Path to your txt file
        filepath = folder_name+"params.txt"
        
        # Open + load
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Modify values
        data["l_w_0"] = lw*np.sqrt(data["a_w"])   # <-- your new value
        data["l_c_0"] = lc*np.sqrt(data["a_c"])   # <-- your new value
        
        # Save changes
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
