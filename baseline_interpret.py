import pickle5 as pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import linalg as LA
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.linalg as scilin

method = "etf_joint_sigma0"
iteration = 1.0
fig_name = method + "_iter" + str(iteration) 
prefix = method + "_num_"

def get_angle(a, b):
    inner_product = (a * b).sum()
    a_norm = a.pow(2).sum().pow(0.5)
    b_norm = b.pow(2).sum().pow(0.5)
    cos = inner_product / (2 * a_norm * b_norm)
    return cos

def get_within_class_covariance(mean_vec_list, feature_dict):
    # feature dimension 512로 fixed
    W = torch.zeros((512, 512))
    total_num = 0

    for klass in list(feature_dict.keys()):
        for i in range(len(feature_dict[klass])):
            W += torch.outer((feature_dict[klass][i] - mean_vec_list[klass]), (feature_dict[klass][i] - mean_vec_list[klass]))
        total_num += len(feature_dict[klass])
    W /= total_num
    return W

def get_between_class_covariance(mean_vec_list, feature_dict):
    B = torch.zeros((512, 512))

    # for global avg calcuation, not just avg mean_vec, feature mean directly (since it is imbalanced dataset)
    total_feature_dict = []
    for key in feature_dict.keys():
        total_feature_dict.extend(feature_dict[key])

    global_mean_vec = torch.mean(torch.stack(total_feature_dict, dim=0), dim=0)

    for klass in list(feature_dict.keys()):
        #B += (mean_vec_list[klass] - global_mean_vec) * (mean_vec_list[klass] - global_mean_vec).T
        B += torch.outer((mean_vec_list[klass] - global_mean_vec), (mean_vec_list[klass] - global_mean_vec))
    B /= len(mean_vec_list)
    return B, global_mean_vec

def get_nc2(mean_vec_list, global_mean_vec):
    M = []
    K = len(list(mean_vec_list.keys()))
    for key in list(mean_vec_list.keys()):
        recentered_mean = mean_vec_list[key] - global_mean_vec
        M.append(recentered_mean / nn.functional.normalize(recentered_mean, p=2.0, dim=0))
    M = torch.stack(M, dim=0)

    nc2_matrix = (torch.matmul(M, M.T) / LA.matrix_norm(torch.matmul(M, M.T))) - ((K-1)**-0.5) * (torch.eye(K) - (1/K)*torch.ones((K,K)))

    return M, K, LA.matrix_norm(nc2_matrix)

def get_nc3(M, A, K):
    nc3_matrix = torch.matmul(A, M.T) / LA.matrix_norm(torch.matmul(A, M.T)) - ((K-1)**-0.5) * (torch.eye(K) - (1/K)*torch.ones((K,K)))
    return LA.matrix_norm(nc3_matrix)

def compute_ETF(W):
    W = W.cuda()
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()

def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    W = W[:K]
    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()
    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H

def compute_Wh_b_relation(W, mu_G, b):
    Wh = torch.mv(W, mu_G.cuda())
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()

nc1_list = []
nc2_list = []
nc3_list = []
etf_list = []
WH_list = []
dist_dict = {}
within_std = {}
between_std = []
between_dist_std = {}

for index in range(100, 50000, 100):
    fc_pickle_name = prefix + str(index) + "_iter" + str(iteration) + "_fc.pickle"
    feature_pickle_name = prefix + str(index) + "_iter" + str(iteration) + "_feature.pickle"
    class_pickle_name = prefix + str(index) + "_iter" + str(iteration) + "_class.pickle"

    with open(fc_pickle_name, 'rb') as f:
        fc_dict = pickle.load(f)

    with open(feature_pickle_name, 'rb') as f:
        feature_dict = pickle.load(f)

    with open(class_pickle_name, 'rb') as f:
        class_dict = pickle.load(f)

    mean_vec_list = {}
    mean_vec_tensor_list = []

    # feature normalize
    mu_G = 0
    num_feature = 0
    for cls in list(feature_dict.keys()):
        feature_dict[cls] = torch.cat(feature_dict[cls]).detach().cpu()
        feature_dict[cls] /= torch.norm(feature_dict[cls], p=2, dim=1, keepdim=True)
        mean_vec_list[cls] = torch.mean(feature_dict[cls], dim=0)
        mu_G += torch.sum(feature_dict[cls], dim=0)
        num_feature += len(feature_dict[cls])
        mean_vec_tensor_list.append(mean_vec_list[cls])

    mean_vec_tensor_list = torch.stack(mean_vec_tensor_list)
    mu_G /= num_feature
    
    #mean_vec_list = [torch.mean(feature_dict[cls], dim=0) for cls in list(feature_dict.keys())]
    print("index", index)
    '''
    ### angle check ###
    for i in range(len(mean_vec_list)):
        for j in range(i+1, len(mean_vec_list)):
            print("i", i, "j", j, "cos", get_angle(mean_vec_list[i], mean_vec_list[j]))
    print()
    '''
    '''
    print("get within covariance")
    print(get_within_class_covariance(mean_vec_list, feature_dict).shape)
    
    print("get between covariance")
    print(get_between_class_covariance(mean_vec_list, feature_dict).shape)
    '''
    
    '''
    ### plot tsne ###
    feature_list = []
    label_list = []
    for key in list(feature_dict.keys()):
        feature_list.extend(feature_dict[key])
        for _ in range(len(feature_dict[key])):
            label_list.append(key)
        
    label_list = np.array(label_list)
    color_list = ["violet", "limegreen", "orange","pink","blue","brown","red","grey","yellow","green"] #cifar10 기준
    tsne_model = TSNE(n_components=2)
    cluster = np.array(tsne_model.fit_transform(torch.stack(feature_list).cpu()))
    plt.figure()
    for i in range(len(list(feature_dict.keys()))):
        idx = np.where(np.array(label_list) == i)
        label = "class" + str(i)
        plt.scatter(cluster[idx[0], 0], cluster[idx[0], 1], marker='.', c=color_list[i], label=label)
        plt.legend()

    tsne_fig_name =  prefix + str(index) + "_iter" + str(iteration) + "_tsne_figure.png"
    plt.savefig(tsne_fig_name)
    ############
    '''
    ### check within std ###
    for feature_key in feature_dict.keys():
        feature_std = torch.mean(torch.std(feature_dict[feature_key], dim=0))
        if feature_key not in within_std.keys():
            within_std[feature_key] = [feature_std]
        else:
            within_std[feature_key].append(feature_std)

    ### check between std ###
    between_std.append(torch.mean(torch.std(torch.stack(list(mean_vec_list.values())),dim=0)).item())
    for i, key_i in enumerate(list(mean_vec_list.keys())):
        for j, key_j in enumerate(list(mean_vec_list.keys())):
            if i>=j:
                continue
            label = str(i) + " and " + str(j)
            #dist_label = ((mean_vec_list[key_i] - mean_vec_list[key_j])**2).sum().sqrt().item()
            dist_label = get_angle(mean_vec_list[key_i], mean_vec_list[key_j])
            if label not in between_dist_std.keys():
                between_dist_std[label] = [dist_label]
            else:
                between_dist_std[label].append(dist_label)


    ### check feature-classifier distance ###
    for feature_key in feature_dict.keys():
        # dist = ((mean_vec_list[feature_key].cpu() - fc_dict[feature_key].cpu())**2).sum().sqrt().item()
        #print("fc_dict shape", fc_dict.shape)
        dist = get_angle(mean_vec_list[feature_key].cpu(), fc_dict[feature_key].cpu())
        if feature_key not in dist_dict.keys():
            dist_dict[feature_key] = [dist]
        else:
            dist_dict[feature_key].append(dist)


    ### check nc1 ###
    W = get_within_class_covariance(mean_vec_list, feature_dict)
    B, global_mean_vec = get_between_class_covariance(mean_vec_list, feature_dict)
    nc1_value = torch.trace(W @ torch.linalg.pinv(B)) / len(mean_vec_list.keys())
    nc1_list.append(nc1_value)

    ### check nc2 ###
    M, K, nc2_value = get_nc2(mean_vec_list, global_mean_vec)
    #nc2_value = compute_ETF(fc_dict)
    nc2_list.append(nc2_value)

    ### check nc3 ###
    nc3_value = get_nc3(M, fc_dict.cpu()[:len(mean_vec_list)], K)
    nc3_list.append(nc3_value)

    ### check ETF (modified nc2) ###
    etf_value = compute_ETF(mean_vec_tensor_list)
    etf_list.append(etf_value)

    ### check W_H_relation (modieifed nc3) ###
    WH_relation_value, H = compute_W_H_relation(fc_dict, mean_vec_list, mu_G)
    WH_list.append(WH_relation_value)


'''
### feature std ###
feature_std_pickle_name = prefix + str(20000) + "_iter" + str(iteration) + "_feature_std.pickle"
with open(feature_std_pickle_name, 'rb') as f:
    feature_std = pickle.load(f)
plt.figure()
plt.title("feature std")
print("len", len(feature_std))
plt.plot(range(len(feature_std)), feature_std)
plt.savefig(fig_name + "_feature_std.png")
'''

### plot within std ###
max_len = 0
for key in within_std.keys():
    max_len = max(max_len, len(within_std[key]))
plt.figure()
plt.title("Within STD per Class")
for key in within_std.keys():
    label = "class" + str(key)
    #print("key", key, "within_std[key]", len(savgol_filter(within_std[key], 7, 3)))
    #print("label", label, "len(within_std[key])", len(within_std[key]))
    try:
        plt.plot(range(max_len)[-len(within_std[key]):], savgol_filter(within_std[key], 15, 3), label=label)
    except:
        pass
plt.legend()
plt.savefig(fig_name + "_within_std_result.png")

### plot between std ###
plt.figure()
plt.title("Between STD per Class")
plt.plot(range(len(between_std)), savgol_filter(between_std, 15, 3), linewidth=3)
plt.savefig(fig_name + "_between_std_result.png")

### plot between distances ###
max_len = 0
for key in list(between_dist_std.keys()):
    max_len = max(max_len, len(between_dist_std[key]))
plt.figure()
plt.title("Distance Between Classes")
for key in between_dist_std.keys():
    labels = [int(i) for i in key.split("and")]
    try:
        plt.plot(range(max_len)[-len(between_dist_std[key]):], savgol_filter(between_dist_std[key], 15, 3), label=key, linewidth=3)
    except:
        pass
plt.legend()
plt.savefig(fig_name + "_class_distances.png")

### plot feature-classifier distance ###
plt.figure()
plt.title("feature-classifier distance")
for key in dist_dict.keys():
    label = "class" + str(key)
    #print("key", key, "within_std[key]", len(savgol_filter(within_std[key], 7, 3)))
    try:
        plt.plot(range(max_len)[-len(dist_dict[key]):], savgol_filter(dist_dict[key], 15, 3), label=label)
    except:
        pass
    
plt.legend()
plt.savefig(fig_name + "_distance_result.png")

### plot nc1 ###
plt.figure()
plt.ylim((0, 1.0))
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("NC1", fontsize=15)
#plt.plot(range(len(nc1_list)), savgol_filter(nc1_list, 15, 3), linewidth='3', color='b')
plt.plot(range(len(nc1_list)), nc1_list, linewidth='3', color='b')
for i in range(1,4):
    plt.axvline(x=i*100, color='r', linestyle='--', linewidth=2)
plt.title("NC1 result", fontsize=20)
plt.savefig(fig_name + "_nc1_result.png")

'''
### plot nc2 ###
plt.figure()
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("NC2", fontsize=15)
plt.plot(range(len(nc2_list)), savgol_filter(nc2_list, 15, 3), linewidth='3', color='b')
for i in range(1,4):
    plt.axvline(x=i*100, color='r', linestyle='--', linewidth=2)
plt.title("NC2 result", fontsize=20)
plt.savefig(fig_name + "_nc2_result.png")

### plot nc3 ###
plt.figure()
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("NC3", fontsize=15)
plt.plot(range(len(nc3_list)), savgol_filter(nc3_list, 15, 3), linewidth='3', color='b')
for i in range(1,4):
    plt.axvline(x=i*100, color='r', linestyle='--', linewidth=2)
plt.title("NC3 result", fontsize=20)
plt.savefig(fig_name + "_nc3_result.png")
'''

### plot ETF (NC2) ###
plt.figure()
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("NC2", fontsize=15)
#plt.plot(range(len(etf_list)), savgol_filter(etf_list, 15, 3), linewidth='3', color='b')
plt.plot(range(len(etf_list)), etf_list, linewidth='3', color='b')
for i in range(1,4):
    plt.axvline(x=i*100, color='r', linestyle='--', linewidth=2)
plt.title("NC2", fontsize=20)
plt.savefig(fig_name + "_etf_result.png")

### plot WH (NC3) ###
plt.figure()
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("NC3", fontsize=15)
plt.plot(range(len(WH_list)), savgol_filter(WH_list, 15, 3), linewidth='3', color='b')
for i in range(1,4):
    plt.axvline(x=i*100, color='r', linestyle='--', linewidth=2)
plt.title("NC3", fontsize=20)
plt.savefig(fig_name + "_WH_result.png")

'''
### plot save ###
plt.savefig(fig_name)
print("figname", fig_name)
'''
