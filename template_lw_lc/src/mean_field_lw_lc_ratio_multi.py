# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 16:38:47 2025

@author: Nemat002
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
import subprocess

def save_object(obj, filename):
    """
    Save an object's attributes to a text file in JSON format.
    Only JSON-compatible attributes (numbers, strings, lists, dicts) will be saved.
    """
    with open(filename, "w") as f:
        json.dump(obj.__dict__, f, indent=4)

def load_object(cls, filename):
    """
    Load attributes from a JSON file and set them on an instance of `cls`.
    Works even if `cls.__init__` takes no arguments.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    obj = cls()  # create instance with default constructor
    for key, value in data.items():
        setattr(obj, key, value)  # assign attributes dynamically
    return obj

class paramsClass:
  # l_w_0 = 0 # how far the effect ON wt cells can reach
  # l_c_0 = 0 # how far the effect ON ca cells can reach
  test = 0
  
class initClass:
  n_w_init = 0
  n_c_init = 0

def params_func():
    params = paramsClass()
    params.char_cell_size = 10 #micron 
    params.l_w_0 = 1.0 * params.char_cell_size # how far the effect ON wt cells can reach
    params.l_c_0 = 1.0 * params.char_cell_size # how far the effect ON ca cells can reach
    
    params.beta_w_unaff = 0.0284
    params.beta_c_unaff = 0.0398
    
    params.beta_w_aff = 0.01
    params.beta_c_aff = 0.0398
    
    return params

def init_func(params, init_n):
    
    init = initClass()
    
    init.n_w_init = init_n.n_w_init
    init.n_c_init = init_n.n_c_init
    
    init.A_w_init = init.n_w_init * params.a_w
    init.A_c_init = init.n_c_init * params.a_c
    
    total_A = init.A_w_init + init.A_c_init
    
    init.r = np.sqrt(total_A/(4*np.pi))
    
    # A = 2*\pi*(r^2)*(1 - cos \theta)
    init.theta_2 = np.arccos( 1 - init.A_c_init / (2 * np.pi * init.r**2) ) # border of wt and c
    
    # split of c
    init.theta_1 = max(0, init.theta_2 - params.l_c_0/init.r)
    init.A_c_unaff = 2 * np.pi * (init.r**2) * (1-np.cos(init.theta_1))
    init.A_c_aff   = init.A_c_init - init.A_c_unaff
    # if init.theta_1>0:
    #     init.A_c_unaff = 2 * np.pi * (init.r**2) * (1-np.cos(init.theta_1))
    #     init.A_c_aff   = init.A_c_init - init.A_c_unaff
    # elif init.theta_1 <= 0:
    #     init.theta_1 = 0
    #     init.A_c_unaff = 0
    #     init.A_c_aff   = init.A_c_init
    # split of c

    # split of wt    
    init.theta_3 = min(np.pi, init.theta_2 + params.l_w_0/init.r)
    init.A_w_unaff = 2 * np.pi * (init.r**2) * (1-np.cos(np.pi - init.theta_3))
    init.A_w_aff   = init.A_w_init - init.A_w_unaff
    # if init.theta_3 < np.pi:
    #     init.A_w_unaff = 2 * np.pi * (init.r**2) * (1-np.cos(np.pi - init.theta_3))
    #     init.A_w_aff   = init.A_w_init - init.A_w_unaff
    # elif init.theta_3 >= np.pi:
    #     init.theta_3 = np.pi
    #     init.A_w_unaff = 0
    #     init.A_w_aff   = init.A_w_init
    # split of wt
    
    return init


def init_numbers_maker():
    
    try:
        n_init_samples = np.loadtxt('n_init_samples.csv', dtype=int, delimiter=',')
        n_org = np.shape(n_init_samples)[0]
    except:
        n_org = 200
        mixed_sample_bank = np.loadtxt("mixed_sample_bank.csv", delimiter=',', dtype=int)
        size = np.shape(mixed_sample_bank)[0]
        sample_indices_mix = np.random.randint(0,size,n_org)
        np.savetxt('sample_indices_mix.csv', X=sample_indices_mix, delimiter=',', fmt='%d')
        
        n_init_samples = np.zeros((n_org,2), dtype=int)
        for org_c in range(n_org):
            n_init_samples[org_c,0] = mixed_sample_bank[sample_indices_mix[org_c],0]
            n_init_samples[org_c,1] = mixed_sample_bank[sample_indices_mix[org_c],1]
        
        np.savetxt('n_init_samples.csv', X=n_init_samples, fmt='%d', delimiter=',')
    
    return n_init_samples

def cost_calc(cost_key):
    
    if cost_key == "w":
        factor_w = 1
        factor_c = 0
    elif cost_key == "c":
        factor_w = 0
        factor_c = 1
    elif cost_key == "both":
        factor_w = 1
        factor_c = 1
        
        
    cost = 0.0
    
    log_w_bar_mod = np.log(np.mean(A_w_mat / A_w_mat[:, [0]], axis=0))
    log_w_bar_mod_err =  (np.std(A_w_mat / A_w_mat[:, [0]], axis=0)/np.sqrt(n_org)) / (np.mean(A_w_mat / A_w_mat[:, [0]], axis=0))
    
    log_w_bar_exp     = np.log(WT_mix_norm_avg)
    log_w_bar_exp_err = WT_mix_norm_err / WT_mix_norm_avg
    
    log_c_bar_mod = np.log(np.mean(A_c_mat / A_c_mat[:, [0]], axis=0))
    log_c_bar_mod_err =  (np.std(A_c_mat / A_c_mat[:, [0]], axis=0)/np.sqrt(n_org)) / (np.mean(A_c_mat / A_c_mat[:, [0]], axis=0))
    
    log_c_bar_exp = np.log(C_mix_norm_avg)
    log_c_bar_exp_err = C_mix_norm_err / C_mix_norm_avg
    
    # plt.figure()
    # plt.errorbar(exp_time, y=log_w_bar_exp, yerr = log_w_bar_exp_err, color='m')
    # plt.errorbar(exp_time, y=log_c_bar_exp, yerr = log_c_bar_exp_err, color='g')
    # plt.errorbar(time, y=log_w_bar_mod, yerr = log_w_bar_mod_err, color='r')
    # plt.errorbar(time, y=log_c_bar_mod, yerr = log_c_bar_mod_err, color='b')
    
    
    log_w_bar_mod_interpolate     = np.interp(exp_time, time, log_w_bar_mod)
    log_w_bar_mod_interpolate_err = np.interp(exp_time, time, log_w_bar_mod_err)
    
    log_c_bar_mod_interpolate     = np.interp(exp_time, time, log_c_bar_mod)
    log_c_bar_mod_interpolate_err = np.interp(exp_time, time, log_c_bar_mod_err)
    
    cost_weights_w = 1/log_w_bar_exp_err**2
    cost_weights_c = 1/log_c_bar_exp_err**2
    
    cost_w = float(np.sum(cost_weights_w[1:] * (log_w_bar_mod_interpolate[1:]-log_w_bar_exp[1:])**2))
    cost_c = float(np.sum(cost_weights_c[1:] * (log_c_bar_mod_interpolate[1:]-log_c_bar_exp[1:])**2))
    
    cost = dict()
    
    cost['w'] = factor_w * cost_w
    cost['c'] = factor_c * cost_c
    cost['tot'] = cost_w + cost_c
    
    return cost
# frame_plotter(0.6,1.4, 2.1,20)


def GD_logger():
    
    try:
        # cost_tot; cost_w; cost_c; b_w; b_c; tau_w; tau_c;
        GD_log = np.loadtxt("GD_log.csv", delimiter=',', dtype=float)
        if GD_log.ndim == 1:
            # Convert to a 1-row matrix
            GD_log = GD_log.reshape(1, -1)
            
        GD_log_new = np.vstack([GD_log, np.zeros((1, GD_log.shape[1]))])
        # GD_log_new[-1,:] = np.array([cost_dict['tot'], cost_dict['w'], cost_dict['c'], b_w, b_c, tau_w, tau_c])
        GD_log_new[-1,:] = np.array([cost_dict['tot'], cost_dict['w'], cost_dict['c'], b_w, b_c])
        np.savetxt('GD_log.csv', X=GD_log_new, delimiter=' , ', fmt='%.8e')
    except:
        # GD_log_new = np.zeros((1,7))
        GD_log_new = np.zeros((1,5))
        # GD_log_new[-1,:] = np.array([cost_dict['tot'], cost_dict['w'], cost_dict['c'], b_w, b_c, tau_w, tau_c])
        GD_log_new[-1,:] = np.array([cost_dict['tot'], cost_dict['w'], cost_dict['c'], b_w, b_c])
        np.savetxt('GD_log.csv', X=GD_log_new, delimiter=' , ', fmt='%.8e')
    
    return 0
    
# exp data
overal_WT_mix = np.loadtxt("exp_data/"+"overal_WT_mix.csv", delimiter=',')
overal_C_mix = np.loadtxt("exp_data/"+"overal_C_mix.csv", delimiter=',')
exp_time   = overal_WT_mix[0,:]
WT_mix_norm_avg = overal_WT_mix[1,:]
WT_mix_norm_err = overal_WT_mix[2,:]
C_mix_norm_avg  = overal_C_mix[1,:]
C_mix_norm_err  = overal_C_mix[2,:]
# exp data

# sim params
os.makedirs("data", exist_ok=True)
dt = 0.1
n_time_steps = 700
time = dt * np.linspace(0,n_time_steps,n_time_steps+1)
np.savetxt("data/"+'time.txt', time, fmt='%.8f', delimiter=',')
# sim params

# temporal delta functions
# b_w = 0.013; tau_w = 20;
# # b_c = 0.1; tau_c = 80;
# # b_w = 0.3; tau_w = 20;
# b_c = 2.0; tau_c = 70;

GD_vals = np.loadtxt("GD_vals.csv", delimiter=',')
# b_w; b_c; tau_w; tau_c;
b_w = GD_vals[0]
b_c = GD_vals[1]
# tau_w = GD_vals[2]
# tau_c = GD_vals[3]

# delta_w = b_w * (1-np.exp(-time/tau_w))
# delta_c = b_c * (1-np.exp(-time/tau_c))

delta_w = b_w * time
delta_c = b_c * time
# temporal delta functions


# reading params
# params = params_func()
# save_object(params, "params.txt")
params = load_object(paramsClass, "params.txt")
# reading params

# n_init values
n_init_samples = init_numbers_maker()
n_org = np.shape(n_init_samples)[0]
# n_init values

# defining matrices for data saving
A_w_mat       = np.zeros((n_org,n_time_steps+1))
A_w_aff_mat   = np.zeros((n_org,n_time_steps+1))
A_w_unaff_mat = np.zeros((n_org,n_time_steps+1))
A_w_v_mat     = np.zeros((n_org,n_time_steps+1)) # part of W, which is visible for C

A_c_mat       = np.zeros((n_org,n_time_steps+1))
A_c_aff_mat   = np.zeros((n_org,n_time_steps+1))
A_c_unaff_mat = np.zeros((n_org,n_time_steps+1))
A_c_v_mat     = np.zeros((n_org,n_time_steps+1)) # part of C, which is visible for W

r_mat    = np.zeros((n_org,n_time_steps+1))
th_1_mat = np.zeros((n_org,n_time_steps+1))
th_2_mat = np.zeros((n_org,n_time_steps+1))
th_3_mat = np.zeros((n_org,n_time_steps+1))
th_v_w_mat = np.zeros((n_org,n_time_steps+1))
th_v_c_mat = np.zeros((n_org,n_time_steps+1))

beta_w_aff_mat = np.zeros((n_org,n_time_steps+1))
beta_c_aff_mat = np.zeros((n_org,n_time_steps+1))
# defining matrices for data saving




# sdfsdf

# for loop on organoids
for org_c in range(n_org):

    
    # init_n = load_object(paramsClass, "init_n.txt")
    
    init_n = initClass()
    init_n.n_w_init = n_init_samples[org_c,0]
    init_n.n_c_init = n_init_samples[org_c,1]
    
    
    init = init_func(params, init_n)
    # save_object(init, "data/init_summary.txt")
    # init = load_object(initClass, "init.txt")
    
    
    
    
    
    
    # read delta_w, delta_c
    # Delta_w_load = np.loadtxt("Delta_W_data.csv", delimiter=',')
    # Delta_c_load = np.loadtxt("Delta_C_data.csv", delimiter=',')
    # Delta_w = np.interp(time, Delta_w_load[0,:], Delta_w_load[1,:])
    # Delta_c = np.interp(time, Delta_c_load[0,:], Delta_c_load[1,:])
    
    # plt.figure()
    # plt.plot(Delta_w_load[0,:], Delta_w_load[1,:], color='m')
    # plt.plot(time, Delta_w, color='r')
    # plt.figure()
    # plt.plot(Delta_c_load[0,:], Delta_c_load[1,:], color='g')
    # plt.plot(time, Delta_c, color='b', linestyle='--')
    
    # delta_w = np.loadtxt("delta_w.txt", delimiter=',')[1,:]
    # delta_c = np.loadtxt("delta_c.txt", delimiter=',')[1,:]
    
    # delta_w = np.loadtxt("delta_w.txt", delimiter=',')
    # delta_c = np.loadtxt("delta_c.txt", delimiter=',')
    # b_w = 0.8; tau_w = 70;
    # b_c = 0.5; tau_c = 50;
    # # b_c = 2.8*11/params.l_c_0; tau_c = 50;
    # delta_w = b_w * (1-np.exp(-time/tau_w))
    # delta_c = b_c * (1-np.exp(-time/tau_c))
    # delta_w = 0.0*np.ones(n_time_steps)
    # delta_c = 1.0*np.ones(n_time_steps+1)
    # np.savetxt("delta_w_guess_0.txt",   delta_w,   fmt='%.8f', delimiter=',')
    # np.savetxt("delta_c_guess_0.txt",   delta_c,   fmt='%.8f', delimiter=',')
    # read delta_w, delta_c
    
    
    
    A_w       = np.zeros(n_time_steps+1)
    A_w_aff   = np.zeros(n_time_steps+1)
    A_w_unaff = np.zeros(n_time_steps+1)
    A_w_v     = np.zeros(n_time_steps+1) # part of W, which is visible for C
    
    A_c       = np.zeros(n_time_steps+1)
    A_c_aff   = np.zeros(n_time_steps+1)
    A_c_unaff = np.zeros(n_time_steps+1)
    A_c_v     = np.zeros(n_time_steps+1) # part of C, which is visible for W
    
    r    = np.zeros(n_time_steps+1)
    th_1 = np.zeros(n_time_steps+1)
    th_2 = np.zeros(n_time_steps+1)
    th_3 = np.zeros(n_time_steps+1)
    th_v_w = np.zeros(n_time_steps+1)
    th_v_c = np.zeros(n_time_steps+1)
    
    beta_w_aff_list = np.zeros(n_time_steps+1)
    beta_c_aff_list = np.zeros(n_time_steps+1)
    
    # G_int = np.zeros(n_time_steps+1) # integral of C^v(theta, t)
    # F_int = np.zeros(n_time_steps+1) # integral of W^v(theta, t)
    # n_theta_discrete = 50
    # A_c_vis  = np.zeros(n_theta_discrete)
    # A_w_vis  = np.zeros(n_theta_discrete)
    # beta_w_th = np.zeros(n_theta_discrete) # beta between th_2 and th_3, which is a function of theta
    # beta_c_th = np.zeros(n_theta_discrete) # beta between th_1 and th_2, which is a function of theta
    
    # beta_w_ll = np.zeros(n_time_steps+1) # ll: last layer
    # beta_c_ll = np.zeros(n_time_steps+1)
    
    A_w[0] = init.A_w_init
    # A_w_aff[0] = init.A_w_aff
    # A_w_unaff[0] = init.A_w_unaff
    
    A_c[0] = init.A_c_init
    # A_c_aff[0] = init.A_c_aff
    # A_c_unaff[0] = init.A_c_unaff
    
    r[0]    = init.r
    th_1[0] = init.theta_1
    th_2[0] = init.theta_2
    th_3[0] = init.theta_3
    th_v_w[0] = min(th_2[0]+params.l_c_0/r[0] , np.pi) #farthest theta IN W, visible for C
    th_v_c[0] = max(th_2[0]-params.l_w_0/r[0] , 0    ) #farthest theta IN C, visible for W
    
    
    os.makedirs("frames", exist_ok=True)
    # frame_plotter(init.theta_1, init.theta_2, init.theta_3, init.r, 0, 'frame_0')
    
    
    PbyP_switch = int(np.loadtxt("PbyP_switch.txt", delimiter=','))
    
    if PbyP_switch: # propagation by proliferation happens. l_w and l_c might change.
        
    
        with open("err.txt", "a") as file:
            file.write("PbyP_switch=1, has a logical problem!")
        sys.exit(1)
            
        # solve
        beta_w_unaff = params.beta_w_unaff
        beta_c_unaff = params.beta_c_unaff
        beta_w_aff = params.beta_w_aff
        beta_c_aff = params.beta_c_aff
        l_w_0 = params.l_w_0
        l_c_0 = params.l_c_0
        l_w = params.l_w_0
        l_c = params.l_c_0
        l_w_list = np.zeros(n_time_steps+1)
        l_c_list = np.zeros(n_time_steps+1)
        l_w_list[0] = l_w_0 
        l_c_list[0] = l_c_0
        # v_l_w = 10 / 20
        for dt_c in range(1, n_time_steps+1):
            
            t = time[dt_c]
            
            # derivs 
            A_w_u_deriv = beta_w_unaff * A_w_unaff[dt_c-1]
            A_c_u_deriv = beta_c_unaff * A_c_unaff[dt_c-1]
            A_w_a_deriv = beta_w_aff * A_w_aff[dt_c-1]
            A_c_a_deriv = beta_c_aff * A_c_aff[dt_c-1]
            # derivs 
            
            # first attempt (they might need correction)
            A_w_unaff[dt_c] = A_w_unaff[dt_c-1] + A_w_u_deriv * dt
            A_c_unaff[dt_c] = A_c_unaff[dt_c-1] + A_c_u_deriv * dt
            A_w_aff[dt_c]   = A_w_aff[dt_c-1] +   A_w_a_deriv * dt
            A_c_aff[dt_c]   = A_c_aff[dt_c-1] +   A_c_a_deriv * dt
            # first attempt (they might need correction)
            
            # area sums , r, and theta2 (The things the dont need correction)
            A_w[dt_c] = A_w_unaff[dt_c] + A_w_aff[dt_c]
            A_c[dt_c] = A_c_unaff[dt_c] + A_c_aff[dt_c]
            r[dt_c]    = ((A_w[dt_c]+A_c[dt_c]) / (4*np.pi))**0.5
            th_2[dt_c] = np.arccos( (A_w[dt_c]-A_c[dt_c]) / (A_w[dt_c]+A_c[dt_c]) )
            # area sums , r, and theta2 (The things the dont need correction)
            
            # calculating widths of affected areas (w and c)
            theta_1_candidate = np.arccos( A_c_aff[dt_c]/(2*np.pi*r[dt_c]**2)+np.cos(th_2[dt_c]))
            width_a_c = r[dt_c] * ( th_2[dt_c] - theta_1_candidate)
            theta_3_candidate = np.arccos(-A_w_aff[dt_c]/(2*np.pi*r[dt_c]**2)+np.cos(th_2[dt_c]))
            width_a_w = r[dt_c] * (-th_2[dt_c] + theta_1_candidate)
            # calculating widths of affected areas (w and c)
            
            # comparison with l_w_0, l_c_0; correction of l_w, l_c;
            # l_c
            if width_a_c>l_c_0: #propagation by proliferation is happening
                l_c = width_a_c
                th_1[dt_c] = theta_1_candidate
                # A_c, A_c_un, and A_c_af are already correct.
            elif width_a_c<l_c_0: #P by P is NOT happening; A_aff and A_unaff must be corrected;
                l_c = l_c_0
                th_1[dt_c] = max( 0.0  , th_2[dt_c] - l_c/r[dt_c])
                A_c_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1-np.cos(th_1[dt_c]))
                A_c_aff[dt_c]   = A_c[dt_c] - A_c_unaff[dt_c]
                #A_c is correct. But A_c_un and A_c_af must be corrected.
            # l_w
            if width_a_w>l_w_0: #propagation by proliferation is happening
                l_w = width_a_w
                th_3[dt_c] = theta_3_candidate
                # A_w, A_w_un, and A_w_af are already correct.
            elif width_a_w<l_w_0: #P by P is NOT happening; A_aff and A_unaff must be corrected;
                l_w = l_w_0
                th_3[dt_c] = min( np.pi, th_2[dt_c] + l_w/r[dt_c])
                A_w_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1+np.cos(th_3[dt_c]))
                A_w_aff[dt_c]   = A_w[dt_c] - A_w_unaff[dt_c]
                #A_w is correct. But A_w_un and A_w_af must be corrected.
            # comparison with l_w_0, l_c_0; correction of l_w, l_c;
            
            l_w_list[dt_c] = l_w
            l_c_list[dt_c] = l_c
            # l_w += v_l_w * dt
            
            # th_1[dt_c] = max( 0.0  , th_2[dt_c] - l_c/r[dt_c])
            # th_3[dt_c] = min( np.pi, th_2[dt_c] + l_w/r[dt_c])
            
            
            
            
            # frame_plotter(th_1[dt_c], th_2[dt_c], th_3[dt_c], r[dt_c], t,  'frame_'+str(dt_c))
        # solve
    
    elif PbyP_switch==0: # propagation by proliferation does not happen. l_w and l_c are fixed.
        
        # solve
        beta_w_unaff = params.beta_w_unaff
        beta_c_unaff = params.beta_c_unaff
        # beta_w_aff = params.beta_w_aff
        # beta_c_aff = params.beta_c_aff
        l_w = params.l_w_0
        l_c = params.l_c_0
        a_w = params.a_w
        a_c = params.a_c
        # v_l_w = 0 / 20
        # l_w_list = np.zeros(n_time_steps+1)
        # l_c_list = np.zeros(n_time_steps+1)
        # l_w_list[0] = l_w_0 
        # l_c_list[0] = l_c_0
        
        # n_1 = np.array([0,0]) #vector to the desired cap_1 (either +k, -k)
        # m_1 = np.array([0,0]) #vector to the smaller counterpart of cap_1 (either +k, -k)
        # n_2 = np.zeros((n_theta_discrete,2)) #vector to the desired cap_2 (theta-dependent) (first component:z, second:x)
        # m_2 = np.zeros((n_theta_discrete,2)) #vector to the smaller counterpart of cap_2 (theta-dependent)
        
        # A_v_dict = dict()
        # A_v_dict[0] = 0.0
        # A_v_dict[1] = 0.0
        # A_v_dict[2] = 0.0
        # A_v_dict[3] = 0.0
        # A_v_dict[4] = 0.0
        
        for dt_c in range(0, n_time_steps):
            
            t = time[dt_c]
            
            # 2pi_r2 = (2*np.pi*r[dt_c]**2)
            # Areas
            A_c_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1-np.cos(th_1[dt_c]))
            A_c_aff[dt_c]   = (2*np.pi*r[dt_c]**2)*(np.cos(th_1[dt_c])-np.cos(th_2[dt_c]))
            A_c_v[dt_c]     = (2*np.pi*r[dt_c]**2)*(np.cos(th_v_c[dt_c])-np.cos(th_2[dt_c]))
            A_w_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1+np.cos(th_3[dt_c]))
            A_w_aff[dt_c]   = (2*np.pi*r[dt_c]**2)*(np.cos(th_2[dt_c])-np.cos(th_3[dt_c]))
            A_w_v[dt_c]     = (2*np.pi*r[dt_c]**2)*(np.cos(th_2[dt_c])-np.cos(th_v_w[dt_c]))
            # Areas
            
            # beta_aff
            # beta_w_aff = beta_w_unaff * np.exp(-delta_w[dt_c] * (a_w/a_c) * (A_c_v[dt_c]/A_w_aff[dt_c]))
            beta_c_aff = beta_c_unaff *   ( 1 + delta_c[dt_c] * (a_c/a_w) * (A_w_v[dt_c]/A_c_aff[dt_c]))
            beta_w_aff = beta_w_unaff * np.exp(-delta_w[dt_c] * (1/a_c) * (A_c_v[dt_c]))
            # beta_c_aff = beta_c_unaff *   ( 1 + delta_c[dt_c] * (1/a_w) * (A_w_v[dt_c]))
            
            beta_w_aff_list[dt_c] = beta_w_aff
            beta_c_aff_list[dt_c] = beta_c_aff
            # beta_aff
            
            ## beta_w (theta, t)
            # calc C^v(theta,t)
            # d_th = (th_3[dt_c]-th_2[dt_c])/n_theta_discrete
            # th_vec = np.linspace(th_2[dt_c]+d_th/2,th_3[dt_c]-d_th/2,n_theta_discrete)
            # th_vec_w = th_vec.copy()
            # th_state = np.zeros(len(th_vec), dtype=int)
            # if l_w/r[dt_c]>np.pi: #C is totally sensed
            #     A_c_vis  = (0.0*A_c_vis + 1.0)*A_c[dt_c]
            # else: #C is partially sensed
            #     ap = l_w/r[dt_c]
            #     th_state[th_vec>(th_2[dt_c]+ap)] = 0 # 0 means no overlap
            #     th_state[th_vec+th_2[dt_c]<ap]   = 1 # 1 means cap1 in totally within cap2
            #     th_state[th_vec+ap<th_2[dt_c]]   = 2 # 2 means cap2 in totally within cap1
            #     th_state[(th_vec<th_2[dt_c]+ap) & (2*np.pi-th_vec<th_2[dt_c]+ap)]   = 3 # 3 means the overlap is a ring (no intersection)
            #     th_state[(th_vec<th_2[dt_c]+ap) & (th_vec>abs(ap-th_2[dt_c])) & (2*np.pi-th_vec>=th_2[dt_c]+ap)]  = 4 # 4 means there is partial overlap and one intersection
                
            #     A_cap1 = A_c[dt_c]
            #     A_cap2 = 2*np.pi*r[dt_c]**2 * (1-np.cos(ap))
                
            #     A_v_dict[0] = 0.0
            #     A_v_dict[1] = A_cap1
            #     A_v_dict[2] = A_cap2
            #     A_v_dict[3] = min(A_cap1,A_cap2) - (4*np.pi*r[dt_c]**2 - max(A_cap1, A_cap2))
            #     A_v_dict[4] = 0.0 # First I put it zero, and then I calculate it.
                
            #     change_points = th_state[1:] != th_state[:-1]
            #     starts = np.r_[0, np.where(change_points)[0] + 1]
            #     ends = np.r_[np.where(change_points)[0] + 1, len(th_state)]
                
            #     A_ov_vec = 0.0 * th_vec.copy()
    
            #     start_1_0 = 0 # start and end of the patch of one intersection (state 4)
            #     end_1_0   = 0
            #     for patch_c in range(len(starts)):
            #         key = int(th_state[starts[patch_c]])
            #         A_ov_vec[starts[patch_c]: ends[patch_c]] = A_v_dict[key]
                    
            #         if key==4:
            #             start_1_0 = int(starts[patch_c])
            #             end_1_0   = int(ends[patch_c])
                
            #     # if there is a path of one intersection
            #     if end_1_0>start_1_0:
            #         caps_ind = np.array((0,0)) # n.m products
                    
            #         alpha_vec = th_vec[start_1_0:end_1_0]
                    
            #         caps_ind = 0 * caps_ind
                    
            #         if th_2[dt_c]<np.pi/2:
            #             caps_ind[0] = 1
            #             zeta_1 = th_2[dt_c]
            #         else:
            #             caps_ind[0] = -1
            #             zeta_1 = np.pi - th_2[dt_c]
            #             alpha_vec = np.pi - alpha_vec
                    
            #         if ap<np.pi/2:
            #             caps_ind[1] = 1
            #             zeta_2 = ap
            #         else:
            #             caps_ind[1] = -1
            #             zeta_2 = np.pi - ap
            #             alpha_vec = np.pi - alpha_vec
                    
            #         ind_sum  = np.sum(caps_ind)
            #         ind_prod = caps_ind[0] * caps_ind[1]
                    
            #         add_fact = A_cap1*(caps_ind[1]<0) + A_cap2*(caps_ind[0]<0) - 4*np.pi*r[dt_c]**2 * (ind_sum<-1)
            #         mul_fact = float(ind_prod)
    
            #         # calculating Omega
            #         gamma_1_vec = np.atan( (np.cos(zeta_2)-np.cos(alpha_vec)*np.cos(zeta_1))/(np.sin(alpha_vec)*np.cos(zeta_1)) )
            #         phi_1_vec = np.acos(np.tan(gamma_1_vec) / np.tan(zeta_1))
            #         eta_1_vec = np.acos(np.sin(gamma_1_vec) / np.sin(zeta_1))
            #         omega_1_vec = 2 * (eta_1_vec - phi_1_vec * np.cos(zeta_1))
                    
            #         gamma_2_vec = np.atan( (np.cos(zeta_1)-np.cos(alpha_vec)*np.cos(zeta_2))/(np.sin(alpha_vec)*np.cos(zeta_2)) )
            #         phi_2_vec = np.acos(np.tan(gamma_2_vec) / np.tan(zeta_2))
            #         eta_2_vec = np.acos(np.sin(gamma_2_vec) / np.sin(zeta_2))
            #         omega_2_vec = 2 * (eta_2_vec - phi_2_vec * np.cos(zeta_2))
                      
            #         omega_vec = omega_1_vec + omega_2_vec
            #         # calculating Omega
                    
            #         A_ov_vec[start_1_0:end_1_0] = add_fact + mul_fact * omega_vec*r[dt_c]**2
            #         A_c_vis = A_ov_vec.copy()
            #     # if there is intersection
                    
            # # calc C^v(theta,t)
            # G_int[dt_c] = sum(A_c_vis*np.sin(th_vec_w)*d_th)*(2*np.pi*r[dt_c]**2)/a_c
            # # beta_w_th = beta_w_unaff - delta_w[dt_c]*A_c_vis/a_c
            # beta_w_th = 0*A_c_vis/a_c
            # # ll: last layer
            # th_w_ll = min(th_3[dt_c], th_2[dt_c] + np.sqrt(a_w) / r[dt_c])
            # th_patch_ll = th_vec[th_vec < th_w_ll]
            # beta_w_ll[dt_c] = np.sum(beta_w_th[th_vec < th_w_ll]*np.sin(th_patch_ll)*d_th) / np.sum(np.sin(th_patch_ll)*d_th)
            # ## beta_w (theta, t)
              
                
            
            
            # ## beta_c (theta, t)
            # # calc W^v(theta,t)
            # d_th = (th_2[dt_c]-th_1[dt_c])/n_theta_discrete
            # th_vec = np.linspace(th_1[dt_c]+d_th/2,th_2[dt_c]-d_th/2,n_theta_discrete)
            # th_vec_c = th_vec.copy()
            # th_state = np.zeros(len(th_vec), dtype=int)
            # if l_c/r[dt_c]>np.pi: #W is totally sensed
            #     A_w_vis  = (0.0*A_w_vis + 1.0)*A_w[dt_c]
            # else: #W is partially sensed
            #     ap = l_c/r[dt_c]
                
            #     th_state[(np.pi-th_vec)> ( (np.pi-th_2[dt_c]) + ap ) ] = 0 # 0 means no overlap
            #     th_state[(np.pi-th_vec)+(np.pi-th_2[dt_c]) < ap]       = 1 # 1 means cap1 in totally within cap2
            #     th_state[(np.pi-th_vec)+ap < (np.pi-th_2[dt_c])]       = 2 # 2 means cap2 in totally within cap1
            #     th_state[((np.pi-th_vec)<(np.pi-th_2[dt_c])+ap) & (2*np.pi-(np.pi-th_vec)<(np.pi-th_2[dt_c])+ap)]   = 3 # 3 means the overlap is a ring (no intersection)
            #     th_state[((np.pi-th_vec)<(np.pi-th_2[dt_c])+ap) & \
            #              ((np.pi-th_vec)>abs(np.pi-th_2[dt_c]-ap)) & \
            #              (2*np.pi-(np.pi-th_vec)>=(np.pi-th_2[dt_c])+ap)]  = 4 # 4 means there is partial overlap and one intersection
                    
                
            #     A_cap1 = A_w[dt_c]
            #     A_cap2 = 2*np.pi*r[dt_c]**2 * (1-np.cos(ap))
                
            #     A_v_dict[0] = 0.0
            #     A_v_dict[1] = A_cap1
            #     A_v_dict[2] = A_cap2
            #     A_v_dict[3] = min(A_cap1,A_cap2) - (4*np.pi*r[dt_c]**2 - max(A_cap1, A_cap2))
            #     A_v_dict[4] = 0.0 # First I put it zero, and then I calculate it.
                
            #     change_points = th_state[1:] != th_state[:-1]
            #     starts = np.r_[0, np.where(change_points)[0] + 1]
            #     ends = np.r_[np.where(change_points)[0] + 1, len(th_state)]
                
            #     A_ov_vec = 0.0 * th_vec.copy()
    
            #     start_1_0 = 0 # start and end of the patch of one intersection (state 4)
            #     end_1_0   = 0
            #     for patch_c in range(len(starts)):
            #         key = int(th_state[starts[patch_c]])
            #         A_ov_vec[starts[patch_c]: ends[patch_c]] = A_v_dict[key]
                    
            #         if key==4:
            #             start_1_0 = int(starts[patch_c])
            #             end_1_0   = int(ends[patch_c])
                
            #     # if there is a path of one intersection
            #     if end_1_0>start_1_0:
            #         caps_ind = np.array((0,0)) # n.m products
                    
            #         alpha_vec = np.pi - th_vec[start_1_0:end_1_0]
                    
            #         caps_ind = 0 * caps_ind
                    
            #         if (np.pi-th_2[dt_c])<np.pi/2:
            #             caps_ind[0] = 1
            #             zeta_1 = (np.pi-th_2[dt_c])
            #         else:
            #             caps_ind[0] = -1
            #             zeta_1 = np.pi - (np.pi-th_2[dt_c])
            #             alpha_vec = np.pi - alpha_vec
                    
            #         if ap<np.pi/2:
            #             caps_ind[1] = 1
            #             zeta_2 = ap
            #         else:
            #             caps_ind[1] = -1
            #             zeta_2 = np.pi - ap
            #             alpha_vec = np.pi - alpha_vec
                    
            #         ind_sum  = np.sum(caps_ind)
            #         ind_prod = caps_ind[0] * caps_ind[1]
                    
            #         add_fact = A_cap1*(caps_ind[1]<0) + A_cap2*(caps_ind[0]<0) - 4*np.pi*r[dt_c]**2 * (ind_sum<-1)
            #         mul_fact = float(ind_prod)
    
            #         # calculating Omega
            #         gamma_1_vec = np.atan( (np.cos(zeta_2)-np.cos(alpha_vec)*np.cos(zeta_1))/(np.sin(alpha_vec)*np.cos(zeta_1)) )
            #         phi_1_vec = np.acos(np.tan(gamma_1_vec) / np.tan(zeta_1))
            #         eta_1_vec = np.acos(np.sin(gamma_1_vec) / np.sin(zeta_1))
            #         omega_1_vec = 2 * (eta_1_vec - phi_1_vec * np.cos(zeta_1))
                    
            #         gamma_2_vec = np.atan( (np.cos(zeta_1)-np.cos(alpha_vec)*np.cos(zeta_2))/(np.sin(alpha_vec)*np.cos(zeta_2)) )
            #         phi_2_vec = np.acos(np.tan(gamma_2_vec) / np.tan(zeta_2))
            #         eta_2_vec = np.acos(np.sin(gamma_2_vec) / np.sin(zeta_2))
            #         omega_2_vec = 2 * (eta_2_vec - phi_2_vec * np.cos(zeta_2))
                      
            #         omega_vec = omega_1_vec + omega_2_vec
            #         # calculating Omega
                    
            #         A_ov_vec[start_1_0:end_1_0] = add_fact + mul_fact * omega_vec*r[dt_c]**2
            #         A_w_vis = A_ov_vec.copy()
            #     # if there is intersection
            
            # # calc W^v(theta,t)
            # F_int[dt_c] = sum(A_w_vis*np.sin(th_vec_c)*d_th)*(2*np.pi*r[dt_c]**2)/a_w
            # beta_c_th = beta_c_unaff + delta_c[dt_c]*A_w_vis/a_w
            # # ll: last layer
            # th_c_ll = max(th_1[dt_c], th_2[dt_c] - np.sqrt(a_c) / r[dt_c])
            # th_patch_ll = th_vec[th_vec > th_c_ll]
            # beta_c_ll[dt_c] = np.sum(beta_c_th[th_vec > th_c_ll]*np.sin(th_patch_ll)*d_th) / np.sum(np.sin(th_patch_ll)*d_th)
            # ## beta_c (theta, t)
            
            # # plot test
            # plt.figure()
            # th_plot_c_unaff = np.linspace(0, th_1[dt_c], 100)
            # plt.plot(th_plot_c_unaff, beta_c_unaff*np.ones(len(th_plot_c_unaff)), color='g')
            # plt.plot(th_vec_c, beta_c_th, color='g')
            # plt.axvline(x=th_2[dt_c], linestyle='--', label=r'$\theta_2$', color='b')
            # plt.axvline(x=th_1[dt_c], linestyle='--', label=r'$\theta_1$', color='r')
            # plt.axvline(x=th_3[dt_c], linestyle='--', label=r'$\theta_3$', color='y')
            # th_plot_w_unaff = np.linspace(th_3[dt_c], np.pi, 100)
            # plt.plot(th_plot_w_unaff, beta_w_unaff*np.ones(len(th_plot_w_unaff)), color='m')
            # plt.plot(th_vec_w, beta_w_th, color='m')
            # plt.plot(th_vec_c, beta_c_unaff*np.ones(len(th_vec_c)), color='g', linestyle='--', alpha=0.5)
            # plt.plot(th_vec_w, beta_w_unaff*np.ones(len(th_vec_w)), color='m', linestyle='--', alpha=0.5)
            # plt.legend()
            # plt.grid()
            # sys.exit()
            # # plot test
            
            
            ## ratio with beta_unaff_w,c
            A_w_deriv = beta_w_unaff * A_w_unaff[dt_c] + beta_w_aff * A_w_aff[dt_c]
            A_c_deriv = beta_c_unaff * A_c_unaff[dt_c] + beta_c_aff * A_c_aff[dt_c]
            ## ratio with beta_unaff_w,c
            
            # A_w_deriv = beta_w_unaff * (A_w_unaff[dt_c] + A_w_aff[dt_c]) - delta_w[dt_c] * G_int[dt_c]
            # A_c_deriv = beta_c_unaff * (A_c_unaff[dt_c] + A_c_aff[dt_c]) + delta_c[dt_c] * F_int[dt_c]
            
            # A_w_deriv = beta_w_unaff * (A_w_unaff[dt_c])
            # A_c_deriv = beta_c_unaff * (A_c_unaff[dt_c])
            
            
            A_w[dt_c+1] = A_w[dt_c] + dt * A_w_deriv
            A_c[dt_c+1] = A_c[dt_c] + dt * A_c_deriv    
            
            r[dt_c+1]    = ((A_w[dt_c+1]+A_c[dt_c+1]) / (4*np.pi))**0.5
            th_2[dt_c+1] = np.arccos( (A_w[dt_c+1]-A_c[dt_c+1]) / (A_w[dt_c+1]+A_c[dt_c+1]) )
            
            # l_w += v_l_w * dt
            
            # l_w_list[dt_c] = l_w
            # l_c_list[dt_c] = l_c
            
            th_1[dt_c+1] = max( 0.0  , th_2[dt_c+1] - l_c/r[dt_c+1])
            th_3[dt_c+1] = min( np.pi, th_2[dt_c+1] + l_w/r[dt_c+1])
            
            th_v_w[dt_c+1] = min(th_2[dt_c+1] + l_c/r[dt_c+1] , np.pi) #farthest theta IN W, visible for C
            th_v_c[dt_c+1] = max(th_2[dt_c+1] - l_w/r[dt_c+1] , 0    ) #farthest theta IN C, visible for W
            
            # th_1[dt_c+1] =  th_2[dt_c+1] - l_c/r[dt_c+1]
            # th_3[dt_c+1] =  th_2[dt_c+1] + l_w/r[dt_c+1]
            
            # INJAAAAAAAAAAAAAAAAAA
            
            # A_w_unaff[dt_c] = 2*np.pi*(r[dt_c]**2) * (1-np.cos(np.pi - th_3[dt_c]))
            # A_w_aff[dt_c]   = A_w[dt_c] - A_w_unaff[dt_c]
            
            # A_c_unaff[dt_c] = 2*np.pi*(r[dt_c]**2) * (1-np.cos( th_1[dt_c] ) )
            # A_c_aff[dt_c]   = A_c[dt_c] - A_c_unaff[dt_c]
            
            
            # frame_plotter(th_1[dt_c], th_2[dt_c], th_3[dt_c], r[dt_c], t,  'frame_'+str(dt_c))
        # solve
    
    
    # last update
    dt_c += 1
    A_c_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1-np.cos(th_1[dt_c]))
    A_c_aff[dt_c]   = (2*np.pi*r[dt_c]**2)*(np.cos(th_1[dt_c])-np.cos(th_2[dt_c]))
    A_w_unaff[dt_c] = (2*np.pi*r[dt_c]**2)*(1+np.cos(th_3[dt_c]))
    A_w_aff[dt_c]   = (2*np.pi*r[dt_c]**2)*(np.cos(th_2[dt_c])-np.cos(th_3[dt_c]))
    
    A_c_v[dt_c]     = (2*np.pi*r[dt_c]**2)*(np.cos(th_v_c[dt_c])-np.cos(th_2[dt_c]))
    A_w_v[dt_c]     = (2*np.pi*r[dt_c]**2)*(np.cos(th_2[dt_c])-np.cos(th_v_w[dt_c]))
    
    # beta_aff
    beta_w_aff = beta_w_unaff * np.exp(-delta_w[dt_c] * (1/a_c) * A_c_v[dt_c])
    beta_c_aff = beta_c_unaff *   ( 1 + delta_c[dt_c] * (a_c/a_w) * (A_w_v[dt_c]/A_c_aff[dt_c]))
    beta_w_aff_list[dt_c] = beta_w_aff
    beta_c_aff_list[dt_c] = beta_c_aff
    # beta_aff
    
    # last update
    
    
    # putting in matrices
    A_w_mat[org_c,:]       = A_w.copy()
    A_w_aff_mat[org_c,:]   = A_w_aff.copy()
    A_w_unaff_mat[org_c,:] = A_w_unaff.copy()
    A_w_v_mat[org_c,:]     = A_w_v.copy()

    A_c_mat[org_c,:]       =  A_c.copy()
    A_c_aff_mat[org_c,:]   =  A_c_aff.copy()
    A_c_unaff_mat[org_c,:] =  A_c_unaff.copy()
    A_c_v_mat[org_c,:]     =  A_c_v.copy()

    r_mat[org_c,:] = r.copy()
    th_1_mat[org_c,:] = th_1.copy()
    th_2_mat[org_c,:] = th_2.copy()
    th_3_mat[org_c,:] = th_3.copy()
    th_v_w_mat[org_c,:] = th_v_w.copy()
    th_v_c_mat[org_c,:] = th_v_c.copy()

    beta_w_aff_mat[org_c,:] = beta_w_aff_list.copy()
    beta_c_aff_mat[org_c,:] = beta_c_aff_list.copy()
    # putting in matrices
    
    # print('********************************')
    # print(org_c)
    # print('********************************')

# for loop on organoids finished




# np.savetxt("data/"+'r.txt', r, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'th_1.txt', th_1, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'th_2.txt', th_2, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'th_3.txt', th_3, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'th_v_w.txt', th_v_w, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'th_v_c.txt', th_v_c, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_w.txt', A_w, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_c.txt', A_c, fmt='%.8f', delimiter=',')

# # np.savetxt("data/"+'G.txt', G_int, fmt='%.8f', delimiter=',')
# # np.savetxt("data/"+'F.txt', F_int, fmt='%.8f', delimiter=',')

# # np.savetxt("data/"+'beta_w_ll.txt', beta_w_ll, fmt='%.8f', delimiter=',')
# # np.savetxt("data/"+'beta_c_ll.txt', beta_c_ll, fmt='%.8f', delimiter=',')

# np.savetxt("data/"+'beta_w_aff_list.txt', beta_w_aff_list, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'beta_c_aff_list.txt', beta_c_aff_list, fmt='%.8f', delimiter=',')

# np.savetxt("data/"+'A_w_aff.txt',   A_w_aff,   fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_w_unaff.txt', A_w_unaff, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_c_aff.txt',   A_c_aff,   fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_c_unaff.txt', A_c_unaff, fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_c_v.txt',   A_c_v,   fmt='%.8f', delimiter=',')
# np.savetxt("data/"+'A_w_v.txt',   A_w_v, fmt='%.8f', delimiter=',')

# # np.savetxt("data/"+'l_w_list.txt', l_w_list, fmt='%.8f', delimiter=',')
# # np.savetxt("data/"+'l_c_list.txt', l_c_list, fmt='%.8f', delimiter=',')


# with open("pp_plotter.py") as f:
#     code = f.read()
# exec(code)

np.savetxt("data/"+'r_mat.txt', r_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'th_1_mat.txt', th_1_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'th_2_mat.txt', th_2_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'th_3_mat.txt', th_3_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'th_v_w_mat.txt', th_v_w_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'th_v_c_mat.txt', th_v_c_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_w_mat.txt', A_w_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_c_mat.txt', A_c_mat, fmt='%.8f', delimiter=',')

np.savetxt("data/"+'beta_w_aff_mat.txt', beta_w_aff_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'beta_c_aff_mat.txt', beta_c_aff_mat, fmt='%.8f', delimiter=',')

np.savetxt("data/"+'A_w_aff_mat.txt',   A_w_aff_mat,   fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_w_unaff_mat.txt', A_w_unaff_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_c_aff_mat.txt',   A_c_aff_mat,   fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_c_unaff_mat.txt', A_c_unaff_mat, fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_c_v_mat.txt',   A_c_v_mat,   fmt='%.8f', delimiter=',')
np.savetxt("data/"+'A_w_v_mat.txt',   A_w_v_mat, fmt='%.8f', delimiter=',')

cost_dict = cost_calc("both")
cost = cost_dict['tot']

GD_logger()



# sdfs
# with open("pp_plotter_multi.py") as f:
#     code = f.read()
# exec(code)

# subprocess.run(["python", "pp_plotter_multi.py"])


