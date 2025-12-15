#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 12:57:06 2025

@author: hossein
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
from scipy.interpolate import UnivariateSpline

def data_saver(matrix, file_name):
    data_to_save =  np.zeros((len(lw_list)+1, len(lc_list)+1))
    data_to_save[1:,0] = lw_list.copy()
    data_to_save[0,1:] = lc_list.copy()
    data_to_save[1:,1:] = matrix.copy()
    np.savetxt("pp_data/"+file_name, X=data_to_save, fmt='%.5e', delimiter=' , ')
    return

lw_list = []
lc_list = []

with open("lw_lc_lists.sh", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("lw_list"):
            values = line.split("=", 1)[1].strip("() ;")
            lw_list = [float(x) for x in values.split()]
        elif line.startswith("lc_list"):
            values = line.split("=", 1)[1].strip("() ;")
            lc_list = [float(x) for x in values.split()]


n_samples = 10


cost_data_avg = np.zeros((len(lw_list), len(lc_list)))
cost_data_err = np.zeros((len(lw_list), len(lc_list)))

deriv_cost_data_avg = np.zeros((len(lw_list), len(lc_list)))
deriv_cost_data_err = np.zeros((len(lw_list), len(lc_list)))

curv_cost_data_avg = np.zeros((len(lw_list), len(lc_list)))
curv_cost_data_err = np.zeros((len(lw_list), len(lc_list)))

b_w_data_avg = np.zeros((len(lw_list), len(lc_list)))
b_w_data_err = np.zeros((len(lw_list), len(lc_list)))

b_c_data_avg = np.zeros((len(lw_list), len(lc_list)))
b_c_data_err = np.zeros((len(lw_list), len(lc_list)))

# tau_w_data_avg = np.zeros((len(lw_list), len(lc_list)))
# tau_w_data_err = np.zeros((len(lw_list), len(lc_list)))

# tau_c_data_avg = np.zeros((len(lw_list), len(lc_list)))
# tau_c_data_err = np.zeros((len(lw_list), len(lc_list)))




# exp data
overal_WT_pure = np.loadtxt("exp_data/"+"overal_WT_pure.csv", delimiter=',')
overal_WT_mix = np.loadtxt("exp_data/"+"overal_WT_mix.csv", delimiter=',')
overal_C_pure = np.loadtxt("exp_data/"+"overal_C_pure.csv", delimiter=',')
overal_C_mix = np.loadtxt("exp_data/"+"overal_C_mix.csv", delimiter=',')

time_data   = overal_WT_pure[0,:]
WT_norm_avg = overal_WT_pure[1,:]
WT_norm_err = overal_WT_pure[2,:]
C_norm_avg = overal_C_pure[1,:]
C_norm_err = overal_C_pure[2,:]

WT_mix_norm_avg = overal_WT_mix[1,:]
WT_mix_norm_err = overal_WT_mix[2,:]
C_mix_norm_avg  = overal_C_mix[1,:]
C_mix_norm_err  = overal_C_mix[2,:]
# exp data

W_mix_fit_smples_coefs = np.loadtxt("W_mix_fit_smples_coefs.csv", delimiter=',')
C_mix_fit_smples_coefs = np.loadtxt("C_mix_fit_smples_coefs.csv", delimiter=',')

for lw_ind in range(len(lw_list)):
    lw = float(lw_list[lw_ind])
    
    for lc_ind in range(len(lc_list)):
        lc = float(lc_list[lc_ind])
        
        
        folder_name = "lw_"+str(lw)+"__lc_"+str(lc)
        
        time = np.loadtxt(folder_name+"/sample_1/data/time.txt", delimiter=',')
        dt = time[1]-time[0]
        
        A_w_stack = np.empty((0,len(time)))
        A_c_stack = np.empty((0,len(time)))
        
        cost_list = []
        cost_list.clear()
        cost_list = []
        
        deriv_cost_list = []
        deriv_cost_list.clear()
        deriv_cost_list = []
        
        curv_cost_list = []
        curv_cost_list.clear()
        curv_cost_list = []
        
        b_w_list = []
        b_w_list.clear()
        b_w_list = []
        
        b_c_list = []
        b_c_list.clear()
        b_c_list = []
        
        # tau_w_list = []
        # tau_w_list.clear()
        # tau_w_list = []
        
        # tau_c_list = []
        # tau_c_list.clear()
        # tau_c_list = []
        
        for sample_c in range(n_samples):
            address = folder_name + "/sample_"+str(sample_c+1)
            GD_log_data = np.loadtxt(address+"/GD_log.csv", delimiter=',')
            cost = GD_log_data[-1, 0]
            cost_list.append(cost)
            
            A_w_sample = np.loadtxt(address+"/data/"+"A_w_mat.txt", delimiter=',')
            A_w_stack = np.vstack([A_w_stack, A_w_sample])
            
            A_c_sample = np.loadtxt(address+"/data/"+"A_c_mat.txt", delimiter=',')
            A_c_stack = np.vstack([A_c_stack, A_c_sample])
            
            
            # cost for first deriv difference
            A_w_norm_sample = A_w_sample / A_w_sample[:, [0]]
            A_w_norm_sample_avg = np.mean(A_w_norm_sample, axis=0)
            spline_w = UnivariateSpline(time, np.log(A_w_norm_sample_avg), s=0)
            f1_w = spline_w.derivative(1)(time)
            f2_w = spline_w.derivative(2)(time)
            curvature_w = f2_w / (1 + f1_w**2)**1.5
            
            fit_coef_sample = np.random.randint(np.shape(W_mix_fit_smples_coefs)[1])
            a = W_mix_fit_smples_coefs[0,fit_coef_sample]
            b = W_mix_fit_smples_coefs[1,fit_coef_sample]
            c = W_mix_fit_smples_coefs[2,fit_coef_sample]
            
            f1_w_exp_fit = 2*a*time+b
            deriv_cost = sum(dt*(f1_w-f1_w_exp_fit)**2)
            deriv_cost_list.append(deriv_cost)
            # cost for first deriv difference
            
            # cost for curv
            fit_coef_sample = np.random.randint(np.shape(W_mix_fit_smples_coefs)[1])
            a = W_mix_fit_smples_coefs[0,fit_coef_sample]
            b = W_mix_fit_smples_coefs[1,fit_coef_sample]
            c = W_mix_fit_smples_coefs[2,fit_coef_sample]
            
            f2_w_exp_fit = 2*a
            fit_curv_w = f2_w_exp_fit / (1+f1_w_exp_fit**2)**1.5
            curv_cost = sum(dt*(curvature_w-fit_curv_w)**2)
            curv_cost_list.append(curv_cost)
            # cost for curv
            
            # b_w, b_c, tau_w, tau_c
            b_w = GD_log_data[-1,3]
            b_c = GD_log_data[-1,4]
            # tau_w = GD_log_data[-1,5]
            # tau_c = GD_log_data[-1,6]
            b_w_list.append(b_w)
            b_c_list.append(b_c)
            # tau_w_list.append(tau_w)
            # tau_c_list.append(tau_c)
            # b_w, b_c, tau_w, tau_c
            
            
        A_w_norm_stack = A_w_stack / A_w_stack[:, [0]]
        A_c_norm_stack = A_c_stack / A_c_stack[:, [0]]
        
        A_w_norm_avg = np.mean(A_w_norm_stack, axis=0)
        A_w_norm_err = np.std(A_w_norm_stack, axis=0)/np.sqrt(np.shape(A_w_norm_stack)[1])
        
        A_c_norm_avg = np.mean(A_c_norm_stack, axis=0)
        A_c_norm_err = np.std(A_c_norm_stack, axis=0)/np.sqrt(np.shape(A_c_norm_stack)[1])
        
        ## plot for this lw , lc
        plt.figure()
        err1 = plt.errorbar(time_data, WT_norm_avg,  yerr=WT_norm_err, fmt='o', color='m', ecolor='m', capsize=2, label='pure WT')
        err2 =plt.errorbar(time_data, C_norm_avg,  yerr=C_norm_err, fmt='o', color='g', ecolor='g', capsize=2, label='pure C')
        err3 = plt.errorbar(time_data, WT_mix_norm_avg, yerr=WT_mix_norm_err, fmt='s', color='m', ecolor='m', capsize=2, label='mixed WT', mfc='none')
        err4 =plt.errorbar(time_data, C_mix_norm_avg, yerr=C_mix_norm_err, fmt='s', color='g', ecolor='g', capsize=2, label='mixed C', mfc='none')
        plt.errorbar(time, y=A_w_norm_avg, yerr = A_w_norm_err, label='w', color='m')
        plt.errorbar(time, y=A_c_norm_avg, yerr = A_c_norm_err, label='c', color='g')
        plt.title(folder_name)
        plt.yscale("log")
        plt.grid()
        plt.legend(fontsize=15)
        plt.xlabel("time", fontsize=15)
        plt.ylabel("N(t)/N(0)", fontsize=15)
        ## plot for this lw , lc
        
        
        # # curvature
        # spline = UnivariateSpline(time, np.log(A_w_norm_avg), s=0)
        # # First and second derivatives
        # f1 = spline.derivative(1)(time)
        # f2 = spline.derivative(2)(time)
        # curvature = f2 / (1 + f1**2)**1.5
        
        # n_discrete = 1000
        # x_smooth = np.linspace(time.min(), time.max(), n_discrete)
        # y_smooth = spline(x_smooth)

        # W_mix_fit_smples_coefs = np.loadtxt("W_mix_fit_smples_coefs.csv", delimiter=',')
        # fit_coef_samples = np.random.randint(0,np.shape(W_mix_fit_smples_coefs)[1],n_samples)
        
        # plt.figure()
        # for i in range(n_samples):
        #     a = W_mix_fit_smples_coefs[0,fit_coef_samples[i]]
        #     b = W_mix_fit_smples_coefs[1,fit_coef_samples[i]]
        #     c = W_mix_fit_smples_coefs[2,fit_coef_samples[i]]
            
        #     fit_prime = 2 * a* x_smooth + b
        #     fit_d_prime = 2*a
        #     fit_curv = fit_d_prime / (1+fit_prime**2)**1.5
            
        #     plt.plot(x_smooth, a*x_smooth**2+b*x_smooth+c, linestyle='--', color='r')
            
        # plt.errorbar(time, y=np.log(A_w_norm_avg), yerr = A_w_norm_err/A_w_norm_avg, label='w', color='m')
        # plt.plot(x_smooth, y_smooth, '-', label='Spline fit', linewidth=2, zorder=10, alpha=0.5)  # spline curve
        # plt.xlabel("t")
        # # plt.ylabel("W(t)/W(0)")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        
        
        # plt.figure()
        # for i in range(n_samples):
        #     a = W_mix_fit_smples_coefs[0,fit_coef_samples[i]]
        #     b = W_mix_fit_smples_coefs[1,fit_coef_samples[i]]
        #     c = W_mix_fit_smples_coefs[2,fit_coef_samples[i]]
            
        #     fit_prime = 2 * a* x_smooth + b
            
        #     plt.plot(x_smooth, 2*a*x_smooth+b, linestyle='--', color='r')
            
        # plt.plot(time, f1, '-', linewidth=2, zorder=10, alpha=0.5)  # spline curve
        # plt.xlabel("t")
        # plt.ylabel("W(t)/W(0)")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        
        
        # plt.figure()
        # for i in range(n_samples):
        #     a = W_mix_fit_smples_coefs[0,fit_coef_samples[i]]
        #     b = W_mix_fit_smples_coefs[1,fit_coef_samples[i]]
        #     c = W_mix_fit_smples_coefs[2,fit_coef_samples[i]]
            
        #     fit_prime = 2 * a* x_smooth + b
        #     fit_d_prime = 2*a
        #     fit_curv = fit_d_prime / (1+fit_prime**2)**1.5
            
        #     plt.plot(x_smooth, fit_curv, linestyle='--', color='r')
            
        # plt.plot(time, curvature, '-', label='Spline fit', linewidth=2, zorder=10, alpha=0.5)  # spline curve
        # plt.xlabel("t")
        # plt.ylabel("curv_W")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        # # curvature
        
        cost_avg = np.mean(cost_list)
        cost_err = np.std(cost_list)/np.sqrt(n_samples-1)
        cost_data_avg[lw_ind, lc_ind] = cost_avg
        cost_data_err[lw_ind, lc_ind] = cost_err
        
        deriv_cost_avg = np.mean(deriv_cost_list)
        deriv_cost_err = np.std(deriv_cost_list)/np.sqrt(n_samples-1)
        deriv_cost_data_avg[lw_ind, lc_ind] = deriv_cost_avg
        deriv_cost_data_err[lw_ind, lc_ind] = deriv_cost_err
        
        curv_cost_avg = np.mean(curv_cost_list)
        curv_cost_err = np.std(curv_cost_list)/np.sqrt(n_samples-1)
        curv_cost_data_avg[lw_ind, lc_ind] = curv_cost_avg
        curv_cost_data_err[lw_ind, lc_ind] = curv_cost_err
        
        b_w_avg = np.mean(b_w_list)
        b_w_err = np.std(b_w_list)/np.sqrt(n_samples-1)
        b_w_data_avg[lw_ind, lc_ind] = b_w_avg
        b_w_data_err[lw_ind, lc_ind] = b_w_err
        
        b_c_avg = np.mean(b_c_list)
        b_c_err = np.std(b_c_list)/np.sqrt(n_samples-1)
        b_c_data_avg[lw_ind, lc_ind] = b_c_avg
        b_c_data_err[lw_ind, lc_ind] = b_c_err
        
        # tau_w_avg = np.mean(tau_w_list)
        # tau_w_err = np.std(tau_w_list)/np.sqrt(n_samples-1)
        # tau_w_data_avg[lw_ind, lc_ind] = tau_w_avg
        # tau_w_data_err[lw_ind, lc_ind] = tau_w_err
        
        # tau_c_avg = np.mean(tau_c_list)
        # tau_c_err = np.std(tau_c_list)/np.sqrt(n_samples-1)
        # tau_c_data_avg[lw_ind, lc_ind] = tau_c_avg
        # tau_c_data_err[lw_ind, lc_ind] = tau_c_err
        
try:
    os.makedirs("pp_data")
except:
    pass
    
    data_saver(cost_data_avg, "cost_data_avg.csv")
    data_saver(cost_data_err, "cost_data_err.csv")
    
    data_saver(deriv_cost_data_avg, "deriv_cost_data_avg.csv")
    data_saver(deriv_cost_data_err, "deriv_cost_data_err.csv")
    
    data_saver(curv_cost_data_avg, "curv_cost_data_avg.csv")
    data_saver(curv_cost_data_err, "curv_cost_data_err.csv")
    
    data_saver(b_w_data_avg, "b_w_data_avg.csv")
    data_saver(b_w_data_err, "b_w_data_err.csv")
    
    data_saver(b_c_data_avg, "b_c_data_avg.csv")
    data_saver(b_c_data_err, "b_c_data_err.csv")
    
    # data_saver(tau_w_data_avg, "tau_w_data_avg.csv")
    # data_saver(tau_w_data_err, "tau_w_data_err.csv")
    
    # data_saver(tau_c_data_avg, "tau_c_data_avg.csv")
    # data_saver(tau_c_data_err, "tau_c_data_err.csv")



# plot cost
plt.figure()
for i in range(len(lc_list)):
    label = 'lc = '+str(lc_list[i])
    plt.errorbar(lw_list, y=cost_data_avg[:,i], yerr = cost_data_err[:,i], label=label)
plt.legend(fontsize=15)
plt.xlabel("l_w", fontsize=15)
plt.ylabel("cost", fontsize=15)
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("cost.PNG", dpi=300)
# plot cost


# plot cost
plt.figure()
for i in range(len(lc_list)):
    label = 'lc = '+str(lc_list[i])
    plt.errorbar(lw_list, y=deriv_cost_data_avg[:,i], yerr = deriv_cost_data_err[:,i], label=label)
plt.legend(fontsize=15)
plt.xlabel("l_w", fontsize=15)
plt.ylabel("deriv_cost_w", fontsize=15)
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("deriv_cost_w.PNG", dpi=300)
# plot cost


# plot cost
plt.figure()
for i in range(len(lc_list)):
    label = 'lc = '+str(lc_list[i])
    plt.errorbar(lw_list, y=curv_cost_data_avg[:,i], yerr = curv_cost_data_err[:,i], label=label)
plt.legend(fontsize=15)
plt.xlabel("l_w", fontsize=15)
plt.ylabel("curv_cost_w", fontsize=15)
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("curv_cost_w.PNG", dpi=300)
# plot cost


# plot b_w
plt.figure()
for i in range(len(lc_list)):
    label = 'lc = '+str(lc_list[i])
    plt.errorbar(lw_list, y=b_w_data_avg[:,i], yerr = b_w_data_err[:,i], label=label)
plt.legend(fontsize=15)
plt.xlabel("l_w", fontsize=15)
plt.ylabel("b_w", fontsize=15)
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("b_w.PNG", dpi=300)
# plot b_w


# plot b_c
plt.figure()
for i in range(len(lc_list)):
    label = 'lc = '+str(lc_list[i])
    plt.errorbar(lw_list, y=b_c_data_avg[:,i], yerr = b_c_data_err[:,i], label=label)
plt.legend(fontsize=15)
plt.xlabel("l_w", fontsize=15)
plt.ylabel("b_c", fontsize=15)
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("b_c.PNG", dpi=300)
# plot b_c


# # plot tau_w
# plt.figure()
# for i in range(len(lc_list)):
#     label = 'lc = '+str(lc_list[i])
#     plt.errorbar(lw_list, y=tau_w_data_avg[:,i], yerr = tau_w_data_err[:,i], label=label)
# plt.legend(fontsize=15)
# plt.xlabel("l_w", fontsize=15)
# plt.ylabel("tau_w", fontsize=15)
# plt.grid()
# plt.tight_layout()
# plt.savefig("tau_w.PNG", dpi=300)
# # plot tau_w

# # plot tau_c
# plt.figure()
# for i in range(len(lc_list)):
#     label = 'lc = '+str(lc_list[i])
#     plt.errorbar(lw_list, y=tau_c_data_avg[:,i], yerr = tau_c_data_err[:,i], label=label)
# plt.legend(fontsize=15)
# plt.xlabel("l_w", fontsize=15)
# plt.ylabel("tau_c", fontsize=15)
# plt.grid()
# plt.tight_layout()
# plt.savefig("tau_c.PNG", dpi=300)
# # plot tau_w

