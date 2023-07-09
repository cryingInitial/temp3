import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings(action='ignore')
dir = 'cifar10'

target_exp_list = ['cifar10/etf_er_resmem_ver3_ce_updated_sigma10_cifar10_non_distill_residual']

label_list = ['ce_non_distill_residual']

target_exp_list = np.array(target_exp_list)
label_list = np.array(label_list)

target_exp_list = np.array(target_exp_list)#[[1,3]]
label_list = np.array(label_list)#[[1,3]]
seed = 1

#if seed == 1:
added_timing = [100, 200, 2400, 3100, 9700, 14500, 18900]
added_timing = added_timing

def print_from_log(exp_name):
    A_auc = []
    A_last = []
    cls_auc = {}
    f = open(f'{exp_name}/seed_{seed}.log', 'r')
    lines = f.readlines()
    for line in lines[11:]:
        if 'Class_Acc' in line:
            list = line.strip().split('|')
            cls = list[2].split()[0]
            acc = list[2].split()[1]
            if cls not in cls_auc.keys():
                cls_auc[cls] = [float(acc)*100]
            else:
                cls_auc[cls].append(float(acc)*100)
    return cls_auc

exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])
cls_auc_list = []
for exp, label in zip(target_exp_list, label_list):
    cls_auc = print_from_log(exp)
    cls_auc_list.append(cls_auc)

max_length = len(cls_auc_list[0][list(cls_auc_list[0].keys())[0]])
for cls in list(cls_auc_list[0].keys()):
    plt.figure()
    plt.title(cls)
    for cls_auc in cls_auc_list:
        plt.plot(range(max_length)[-len(cls_auc[cls]):], savgol_filter(cls_auc[cls], 5, 3), label=cls, linewidth=1.0)
    for i in added_timing:
        plt.axvline(x=i/100, color='r', linestyle='--', linewidth=0.5)
    figure_name = cls+".png" 
    plt.legend()
    plt.savefig(figure_name)


