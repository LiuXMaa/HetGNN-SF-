import networkx as nx
import numpy as np
import scipy
import pickle
import torch


def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')#读取PAP文件
    adjlist00 = [line.strip() for line in in_file]#list文件，长度4019，PS边类型
    adjlist00 = adjlist00[3:]#这里显示的是4019行，连接的节点数
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')#读取PSP文件
    adjlist01 = [line.strip() for line in in_file]#list 长度4019
    adjlist01 = adjlist01[3:]#长度4019
    #每个都是节点数连接其他节点，4019行
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')#读取PAP文件中节点索引
    idx00 = pickle.load(in_file)#字典，经过元路径PAP所连接的数组
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')#读取PSP文件中节点索引
    idx01 = pickle.load(in_file)#字典，经过元路径PSP所连接的数组
    in_file.close()
    emb = np.load(prefix + '/metapath2vec_emb.npy')#加载经过metapath2vec的embedding,这里是所有节点的嵌入


    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()#节点类型1的特征，7167行4000列
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()#节点类型2的特征，60行4000列

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()#这里加载的是原始图
    type_mask = np.load(prefix + '/node_types.npy')#行向量，11246，对应的00000,11111,222222
    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引
    return [adjlist00, adjlist01], \
           [idx00, idx01], \
           [features_0, features_1, features_2], emb, \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx   #返回的元路径1和元路径2下的图及其索引，返回三种类型节点特征和所有节点的嵌入


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    # 0 for movies, 1 for directors, 2 for actors
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    emb = np.load(prefix + '/metapath2vec_emb.npy')

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    # Using MAM+MDM to define relations between movies.
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    adjM = adjM.toarray()
    adjM = torch.FloatTensor(adjM).to(device)
    m_mask = np.where(type_mask == 0)[0]
    d_mask = np.where(type_mask == 1)[0]
    a_mask = np.where(type_mask == 2)[0]
    a_mask = torch.LongTensor(a_mask).to(device)
    m_mask = torch.LongTensor(m_mask).to(device)
    d_mask = torch.LongTensor(d_mask).to(device)

    adjM[m_mask, :][:, m_mask] = torch.mm(adjM[m_mask, :][:, a_mask], adjM[a_mask, :][:, m_mask])
    adjM[m_mask, :][:, m_mask] = adjM[m_mask, :][:, m_mask] + torch.mm(adjM[m_mask, :][:, d_mask], adjM[d_mask, :][:, m_mask])
    adjM = adjM.data.cpu().numpy()
    torch.cuda.empty_cache()

    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2], emb,\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    # 0 for authors, 1 for papers, 2 for terms, 3 for conferences
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#（4057,334）
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()#（14328,4231）
    features_2 = np.load(prefix + '/features_2.npy')#(7723,50)
    features_3 = np.eye(20, dtype=np.float32)

    emb = np.load(prefix + '/metapath2vec_emb.npy')

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    # Using PAP to define relations between papers.
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    adjM = adjM.toarray()
    adjM = torch.FloatTensor(adjM).to(device)
    a_mask = np.where(type_mask == 0)[0]
    p_mask = np.where(type_mask == 1)[0]
    a_mask = torch.LongTensor(a_mask).to(device)
    p_mask = torch.LongTensor(p_mask).to(device)
    adjM[p_mask, :][:, p_mask] = torch.mm(adjM[p_mask, :][:, a_mask], adjM[a_mask, :][:, p_mask])
    adjM = adjM.data.cpu().numpy()
    torch.cuda.empty_cache()

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3], emb,\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx
