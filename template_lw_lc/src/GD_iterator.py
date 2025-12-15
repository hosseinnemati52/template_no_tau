# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:55:30 2025

@author: nemat002
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import ast
import matplotlib.animation as animation
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress
import json
import os
import shutil
import subprocess

def reset_and_fill_folder(target_dir, sources):
    """
    target_dir: str
        Path to the folder you want to clear and repopulate.
    sources: list of paths
        Each path can be a file OR a folder to be copied into target_dir.
    """

    # 1. Remove the folder if it exists
    if os.path.exists(target_dir):
        # shutil.rmtree(target_dir)
        a =1

    # 2. Recreate it empty
    os.makedirs(target_dir, exist_ok=True)

    # 3. Copy each source (file or folder) into the target_dir
    for src in sources:
        if os.path.isdir(src):
            # copy folder → target_dir/src_name
            dst = os.path.join(target_dir, os.path.basename(src))
            shutil.copytree(src, dst)
        else:
            # copy file → target_dir/
            shutil.copy2(src, target_dir)

# initial random values
b_w = 10**np.random.uniform(-3,-1.5)
b_c = 10**np.random.uniform(-3,-1.5)
# tau_w = int(np.random.uniform(2000,5000))
# tau_c = int(np.random.uniform(2000,5000))
# np.savetxt('GD_vals.csv', X=[b_w, b_c, tau_w, tau_c], delimiter=' , ', fmt='%.5e')
np.savetxt('GD_vals.csv', X=np.array([[b_w, b_c]]), delimiter=' , ', fmt='%.8e')
# initial random values

# initial run
subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
# initial run

# reading the history
GD_LOG = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)
if GD_LOG.ndim == 1:
    # Convert to a 1-row matrix
    GD_LOG = GD_LOG.reshape(1, -1)
# b_w; b_c; tau_w; tau_c;
GD_vals = GD_LOG[-1,3:]
b_w = GD_vals[0]
b_c = GD_vals[1]
# tau_w = GD_vals[2]
# tau_c = GD_vals[3]
# reading the history


# make folder GD_temp
target = "GD_temp"
folders_to_copy = ["exp_data"]
files_to_copy = [
    "frame_switch.txt",
    "GD_vals.csv",
    "mean_field_lw_lc_ratio_multi.py",
    "pp_plotter_multi.py", 
    "mixed_sample_bank.csv",
    "n_init_samples.csv",
    "params.txt",
    "PbyP_switch.txt", 
    "sample_indices_mix.csv"
]
sources = folders_to_copy + files_to_copy
reset_and_fill_folder(target, sources)
# make folder GD_temp


converge_cond = 0
n_iter_check = 20
conv_thresh = 0.005
sequence_check = dict()
sequence_check['bw'] = 1.0*np.ones(n_iter_check)
sequence_check['bc'] = 1.0*np.ones(n_iter_check)
# sequence_check['tau_w'] = 1.0*np.ones(n_iter_check)
# sequence_check['tau_c'] = 1.0*np.ones(n_iter_check)

# GD variables
log_bw = np.log(b_w)
log_bc = np.log(b_c)
# log_tau_w = np.log(tau_w)
# log_tau_c = np.log(tau_c)
# GD variables

# GD params
grad = dict()
grad['log_bw'] = 0.0
grad['log_bc'] = 0.0
# grad['log_tau_w'] = 0.0
# grad['log_tau_c'] = 0.0

log_bw_factor = 1.0
log_bc_factor = 1.0
# log_tau_w_factor = 1.0
# log_tau_c_factor = 1.0

w = 0.5  # update weight

pos_neg_perc = 0.005
learning_rate = 5e-3
# GD params

bw_list = [b_w]
bc_list = [b_c]
# tau_w_list = [tau_w]
# tau_c_list = [tau_c]

mother_dir = os.getcwd()

counter = 0
while (not converge_cond):
    
    ## grad evaluation
    # to log_bw
    #pos
    log_bw_pos = log_bw + pos_neg_perc*abs(log_bw)
    b_w_pos = np.exp(log_bw_pos)
    # GD_vals_temp = np.array([[b_w_pos, b_c, tau_w, tau_c]])
    GD_vals_temp = np.array([[b_w_pos, b_c]])
    np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.8e')
    os.chdir("GD_temp")   # go into daughter folder
    subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    try:
        cost_b_w_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    except IndexError:
        cost_b_w_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    os.chdir(mother_dir)
    #pos
    #neg
    log_bw_neg = log_bw - pos_neg_perc*abs(log_bw)
    b_w_neg = np.exp(log_bw_neg)
    # GD_vals_temp = np.array([[b_w_neg, b_c, tau_w, tau_c]])
    GD_vals_temp = np.array([[b_w_neg, b_c]])
    np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.8e')
    os.chdir("GD_temp")   # go into daughter folder
    subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    try:
        cost_b_w_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    except IndexError:
        cost_b_w_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    os.chdir(mother_dir)
    #neg
    grad['log_bw'] = (cost_b_w_pos-cost_b_w_neg)/(log_bw_pos-log_bw_neg)
    # to log_bw
    
    # to log_bc
    #pos
    log_bc_pos = log_bc + pos_neg_perc*abs(log_bc)
    b_c_pos = np.exp(log_bc_pos)
    # GD_vals_temp = np.array([[b_w, b_c_pos, tau_w, tau_c]])
    GD_vals_temp = np.array([[b_w, b_c_pos]])
    np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.8e')
    os.chdir("GD_temp")   # go into daughter folder
    subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    try:
        cost_b_c_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    except IndexError:
        cost_b_c_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    os.chdir(mother_dir)
    #pos
    #neg
    log_bc_neg = log_bc - pos_neg_perc*abs(log_bc)
    b_c_neg = np.exp(log_bc_neg)
    # GD_vals_temp = np.array([[b_w, b_c_neg, tau_w, tau_c]])
    GD_vals_temp = np.array([[b_w, b_c_neg]])
    np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.8e')
    os.chdir("GD_temp")   # go into daughter folder
    subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    try:
        cost_b_c_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    except IndexError:
        cost_b_c_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    os.chdir(mother_dir)
    #neg
    grad['log_bc'] = (cost_b_c_pos-cost_b_c_neg)/(log_bc_pos-log_bc_neg)
    # to log_bc
    
    # to log_tau_w
    #pos
    # log_tau_w_pos = log_tau_w + pos_neg_perc*abs(log_tau_w)
    # tau_w_pos = np.exp(log_tau_w_pos)
    # GD_vals_temp = np.array([[b_w, b_c, tau_w_pos, tau_c]])
    # np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.5e')
    # os.chdir("GD_temp")   # go into daughter folder
    # subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    # try:
    #     cost_tau_w_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    # except IndexError:
    #     cost_tau_w_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    # os.chdir(mother_dir)
    # #pos
    # #neg
    # log_tau_w_neg = log_tau_w - pos_neg_perc*abs(log_tau_w)
    # tau_w_neg = np.exp(log_tau_w_neg)
    # GD_vals_temp = np.array([[b_w, b_c, tau_w_neg, tau_c]])
    # np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.5e')
    # os.chdir("GD_temp")   # go into daughter folder
    # subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    # try:
    #     cost_tau_w_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    # except IndexError:
    #     cost_tau_w_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    # os.chdir(mother_dir)
    #neg
    # grad['log_tau_w'] = (cost_tau_w_pos-cost_tau_w_neg)/(log_tau_w_pos-log_tau_w_neg)
    # grad['log_tau_w'] = 0
    # to log_tau_w
    
    # to log_tau_c
    #pos
    # log_tau_c_pos = log_tau_c + pos_neg_perc*abs(log_tau_c)
    # tau_c_pos = np.exp(log_tau_c_pos)
    # GD_vals_temp = np.array([[b_w, b_c, tau_w, tau_c_pos]])
    # np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.5e')
    # os.chdir("GD_temp")   # go into daughter folder
    # subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    # try:
    #     cost_tau_c_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    # except IndexError:
    #     cost_tau_c_pos = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    # os.chdir(mother_dir)
    # #pos
    # #neg
    # log_tau_c_neg = log_tau_c - pos_neg_perc*abs(log_tau_c)
    # tau_c_neg = np.exp(log_tau_c_neg)
    # GD_vals_temp = np.array([[b_w, b_c, tau_w, tau_c_neg]])
    # np.savetxt('GD_temp/'+'GD_vals.csv', X=GD_vals_temp, delimiter=' , ', fmt='%.5e')
    # os.chdir("GD_temp")   # go into daughter folder
    # subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    # try:
    #     cost_tau_c_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[-1,0]
    # except IndexError:
    #     cost_tau_c_neg = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)[0]
    # os.chdir(mother_dir)
    # #neg
    # grad['log_tau_c'] = (cost_tau_c_pos-cost_tau_c_neg)/(log_tau_c_pos-log_tau_c_neg)
    # grad['log_tau_c'] = 0
    # to log_tau_c
    ## grad evaluation
        
    
    ## update variables
    delta_log_bw = - w * learning_rate * log_bw_factor * grad['log_bw']
    delta_log_bc = - w * learning_rate * log_bc_factor * grad['log_bc']
    # delta_log_tau_w = - w * learning_rate * log_tau_w_factor * grad['log_tau_w']
    # delta_log_tau_c = - w * learning_rate * log_tau_c_factor * grad['log_tau_c']
    
    log_bw += delta_log_bw
    log_bc += delta_log_bc
    # log_tau_w += delta_log_tau_w
    # log_tau_c += delta_log_tau_c
    
    b_w  = np.exp(log_bw)
    b_c  = np.exp(log_bc)
    # tau_w  = np.exp(log_tau_w)
    # tau_c  = np.exp(log_tau_c)
    
    
    # GD_vals = np.array([[b_w, b_c, tau_w, tau_c]])
    GD_vals = np.array([[b_w, b_c]])
    np.savetxt('GD_vals.csv', X=GD_vals, delimiter=' , ', fmt='%.8e')
    
    bw_list.append(b_w)
    bc_list.append(b_c)
    # tau_w_list.append(tau_w)
    # tau_c_list.append(tau_c)
    ## update variables
    
    
    
    ## updating sequences
    sequence_check['bw'][:-1] = sequence_check['bw'][1:]
    sequence_check['bw'][-1]  = b_w # new
    sequence_check['bc'][:-1] = sequence_check['bc'][1:]
    sequence_check['bc'][-1]  = b_c # new
    # sequence_check['tau_w'][:-1] = sequence_check['tau_w'][1:]
    # sequence_check['tau_w'][-1]  = tau_w # new
    # sequence_check['tau_c'][:-1] = sequence_check['tau_c'][1:]
    # sequence_check['tau_c'][-1]  = tau_c # new
    ## updating sequences
    
    ## converge condition
    bw_mat_check = np.abs(sequence_check['bw'][:,None]-sequence_check['bw'][None,:]) \
        /np.abs(sequence_check['bw'][:,None])
    bc_mat_check = np.abs(sequence_check['bc'][:,None]-sequence_check['bc'][None,:]) \
        /np.abs(sequence_check['bc'][:,None])
    
    converge_cond = \
        np.max(bw_mat_check) < conv_thresh and \
        np.max(bc_mat_check) < conv_thresh
    # converge_cond = \
    # max(abs(np.diff(sequence_check['bw'])/sequence_check['bw'][:-1])) < conv_thresh and \
    # max(abs(np.diff(sequence_check['bc'])/sequence_check['bc'][:-1])) < conv_thresh #and \
    # max(abs(np.diff(sequence_check['tau_w'])/sequence_check['tau_w'][:-1])) < conv_thresh and \
    # max(abs(np.diff(sequence_check['tau_c'])/sequence_check['tau_c'][:-1])) < conv_thresh
    ## converge condition
    
    # new run
    subprocess.run(["python", "mean_field_lw_lc_ratio_multi.py"])  # run code here
    # new run
    
    counter += 1
    
    print("********************************")
    print("iteration: "+str(counter))
    print("********************************")
    
print("##################")
print("CONVERGENCE: pp running!")
subprocess.run(["python", "pp_plotter_multi.py"])
print("##################")
    





