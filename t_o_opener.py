#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:12:45 2025

@author: hossein
"""

import json
import numpy as np
import os
import subprocess
import time

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

orig_dir = os.getcwd()

for lw_ind in range(len(lw_list)):
    lw = lw_list[lw_ind]
    
    for lc_ind in range(len(lc_list)):
        lc = lc_list[lc_ind]
        
        
        folder_name = orig_dir+"/lw_"+str(lw)+"__lc_"+str(lc)
        
        script = "do_all.sh"

        # Save original working directory
        
        try:
            os.chdir(folder_name)
        
            title = "/lw_"+str(lw)+"__lc_"+str(lc)
            # Run script in a new terminal window
            subprocess.run([
                "gnome-terminal",
                "--title=" + title,
                "--", "bash", "-c",
                f"./{script}; exec bash"
            ])
        
        finally:
            # Return to the original directory
            os.chdir(orig_dir)
            
        time.sleep(1)
