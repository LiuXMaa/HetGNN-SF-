import argparse
import sys


argv = sys.argv



def ACM_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.75)
    parser.add_argument('--feat_drop', type=float, default=0.5)
    parser.add_argument('--attn_drop', type=float, default=0.6)
    parser.add_argument('--lam', type=float, default=0.45)
    
    args, _ = parser.parse_known_args()
    return args


def IMDB_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.6)#0.6
    parser.add_argument('--feat_drop', type=float, default=0.55)
    parser.add_argument('--attn_drop', type=float, default=0.65)
    parser.add_argument('--lam', type=float, default=0.4)

    args, _ = parser.parse_known_args()
    return args




def DBLP_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=54)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.6)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    return args





def set_params(dataset):
    if dataset == "ACM":
        args = ACM_params()
    elif dataset == "DBLP":
        args = DBLP_params()
    elif dataset == "IMDB":
        args = IMDB_params()

    return args
