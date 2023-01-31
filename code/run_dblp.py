import numpy
import torch
from utils import set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
import numpy as np
from scipy import sparse
import scipy
import torch as th
from scipy import sparse
import scipy.sparse as sp
import dgl
from util.tools import evaluate_results_nc
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
args = set_params('DBLP')
if torch.cuda.is_available():
    device = torch.device("cpu")
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = 'DBLP'

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

data_path = '../data/DBLP_L/'
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def load_data1():
    feature =scipy.sparse.load_npz(data_path + 'a_feature.npz').toarray()
    label=np.zeros((4057,4))
    idx_train = []
    idx_val = []
    idx_test= []
    f3 = open(data_path + 'label_train.dat', 'r', encoding='utf-8')
    for line in f3.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_train.append(a)
    f4 = open(data_path + 'label_val.dat', 'r', encoding='utf-8')
    for line in f4.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_val.append(a)
    f5 = open(data_path + 'label_test.dat', 'r', encoding='utf-8')
    for line in f5.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        a = int(a)
        c = int(c)
        label[a][c] = 1
        idx_test.append(a)

    path = "../data/DBLP_L/"
    pos = sp.load_npz(path + "pos_dblp_apcpa_aptpa700.npz")
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    return feature, label, idx_train, idx_val, idx_test,pos
features, labels, idx_train, idx_val, idx_test,pos = load_data1()
def get_A(data_path):
    # adj=scipy.sparse.load_npz(data_path + 'adjM.npz').toarray()
    # pa = np.zeros((14328,4057))
    # pt = np.zeros((14328, 7723))
    # pv = np.zeros((14328, 20))
    # for i in range(4057,4057+14328):
    #     for j in range(len(adj)):
    #         if adj[i][j]!=0:
    #             if j<4057:
    #                 pa[i-4057][j]=1
    #             elif j>=(4057+14328) and j<(4057+14328+7723):
    #                 pt[i-4057][j-4057-14328]=1
    #             elif j>=(4057+14328+7723):
    #                 pv[i-4057][j-4057-14328-7723]=1
    # ap=pa.T
    # tp=pt.T
    # vp=pv.T
    # apa=ap.dot(pa)
    # sp.save_npz(data_path + 'apa.npz', scipy.sparse.csr_matrix(apa))
    # apt = ap.dot(pt)
    # aptp = apt.dot(tp)
    # aptpa = aptp.dot(pa)
    # sp.save_npz(data_path + 'aptpa.npz', scipy.sparse.csr_matrix(aptpa))
    # apv = ap.dot(pv)
    # apvp = apv.dot(vp)
    # apvpa = apvp.dot(pa)
    # sp.save_npz(data_path + 'apvpa.npz', scipy.sparse.csr_matrix(apvpa))
    apa = scipy.sparse.load_npz(data_path + 'apa.npz').toarray()
    print('apa_max:',apa[np.unravel_index(np.argmax(apa, axis=None), apa.shape)])
    apvpa = scipy.sparse.load_npz(data_path + 'apvpa.npz').toarray()
    print('apvpa_max:', apa[np.unravel_index(np.argmax(apvpa, axis=None), apvpa.shape)])
    aptpa = scipy.sparse.load_npz(data_path + 'aptpa.npz').toarray()
    print('aptpa_max:', apa[np.unravel_index(np.argmax(aptpa, axis=None), aptpa.shape)])
    aa=4*apvpa+1*apa+aptpa#+aptpa
    print('aa_max:', apa[np.unravel_index(np.argmax(apa, axis=None), apa.shape)])
    return aa
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx
def train():
    features, labels, idx_train, idx_val, idx_test,pos = load_data1()
    feats_dim =features.shape[1]
    features = torch.from_numpy(features)
    features = features.float()
    labels = torch.LongTensor(np.where(labels)[1])

    adj= get_A(data_path)#adj作为GCN下的矩阵
    adj[adj <40 ] = 0#GCN去掉条边的链接,全部的话改成13，apa和apvpa的话改成7
    adj = torch.from_numpy(adj).type(torch.FloatTensor)
    adj = F.normalize(adj, dim=1, p=2)
    adj = scipy.sparse.csr_matrix(adj)
    e = torch.tensor(adj.data).type(torch.FloatTensor)
    g1 = dgl.DGLGraph(adj)  # 这里是GCN的

    A = get_A(data_path) # A为GAT下的矩阵
    A[A <40] = 0#GAT去掉6条边的链接,全部的话改成13，apa和apvpa的话改成7
    A = torch.from_numpy(A).type(torch.FloatTensor)
    A = normalize(A)
    A = np.array(A.tolist())
    adjM2 = scipy.sparse.csr_matrix(A)
    g = dgl.DGLGraph(adjM2)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)#这里的是GAT的

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = HeCo(args.hidden_dim, feats_dim, args.feat_drop, args.attn_drop,
                     args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(features, g1,e,g,pos)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_'+own_str+'.pkl'))
    model.eval()
    os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(features, g1,e,g)#这里我们返回的是GAT路径下的结果
    #=============================以下代码为可视化的效果
    # Y = labels.cpu().numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(embeds.cpu().data.numpy())
    # color_idx = {}
    # for i in range(4057):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#FF0000', s=15, alpha=1)
    # plt.legend()
    # plt.savefig("DBLP分类图" +  ".png", dpi=1000, bbox_inches='tight')
    # plt.show()
    # # =======================================================一直到这里↑↑都是可视化
    #进行测试
    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeds[idx_test].cpu().data.numpy(), labels[idx_test].cpu().numpy(), 3)
    #保存嵌入
    f = open("./embeds/" + args.dataset + str(args.turn) + ".pkl", "wb")
    pkl.dump(embeds.cpu().data.numpy(), f)
    f.close()

if __name__ == '__main__':
    train()
