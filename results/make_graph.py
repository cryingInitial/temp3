import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings(action='ignore')
dir = 'cifar10'

target_exp_list = ['cifar10/etf_er_resmem_not_pre_trained_sigma10_cifar10_iter_1_temp_0.7_softmax_top_k_3', \
        'cifar10/etf_er_resmem_ver3_non_distill_not_pre_trained_sigma10_real_cifar10_iter_1_knn_sigma_0.7_top_k_3_softmax_temp_1.0_loss_ce', \
        'cifar10/etf_er_resmem_ver3_non_distill_not_pre_trained_sigma10_real_cifar10_iter_1_knn_sigma_0.7_top_k_3_softmax_temp_1.0_loss_dr', \
        'cifar10/etf_er_resmem_ver3_distill_not_pre_trained_sigma10_real_cifar10_iter_1_knn_sigma_0.7_distill_coeff_0.99_distill_beta_0.1_top_k_3_softmax_temp_1.0_loss_ce_classwise_difference_ver2_threshold_0.5', \
        'cifar10/etf_er_resmem_ver3_distill_not_pre_trained_sigma10_real_cifar10_iter_1_knn_sigma_0.7_distill_coeff_0.99_distill_beta_0.1_top_k_3_softmax_temp_1.0_loss_ce_classwise_difference_ver2_threshold_0.0', \
        'cifar10/etf_er_resmem_ver3_distill_not_pre_trained_sigma10_real_cifar10_iter_1_knn_sigma_0.7_distill_coeff_0.99_distill_beta_0.1_top_k_3_softmax_temp_1.0_loss_dr_classwise_difference_ver2_threshold_0.0'
                   ]

label_list = ['ver2_non_distill', 'non_distill_ce', 'non_distill_dr', 'beta_0.1_thr_0.5_diff_ver2_ce', 'beta_0.1_thr_0.0_diff_ver2_ce', 'beta_0.1_thr_0.0_diff_ver2_dr']

target_exp_list = np.array(target_exp_list)[[1,5]]
label_list = np.array(label_list)[[1,5]]

target_exp_list = np.array(target_exp_list)#[[1,3]]
label_list = np.array(label_list)#[[1,3]]
seed = 2

#if seed == 1:
added_timing = [100, 200, 2400, 3100, 9700, 14500, 18900]
added_timing = added_timing

def print_from_log(exp_name):
    A_auc = []
    A_last = []
    f = open(f'{exp_name}/seed_{seed}.log', 'r')
    lines = f.readlines()
    for line in lines[11:]:
        if 'test_acc' in line:
            list = line.split('|')
            A_auc.append(float(list[3].split(' ')[2])*100)
    return A_auc

exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])

for exp, label in zip(target_exp_list, label_list):
    A_auc = print_from_log(exp)
    plt.plot(range(len(A_auc)), savgol_filter(A_auc, 5, 3), label=label, linewidth=1.0)
for i in added_timing:
    plt.axvline(x=i/100, color='r', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig('resulgs.png')


