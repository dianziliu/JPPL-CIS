from math import log2
from random import shuffle
import numpy as np
import pandas as pd
import joblib
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm,tqdm_pandas
from collections import defaultdict
from sklearn.metrics import precision_score,recall_score

import torch

"""
需要注意到，NDCG,MAP存在多种计算方式
"""


def AP(K, scores, ids, method=0):
    """
    scores:[iId,score,rating]
    ids:用户的测试集
    """
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]] 
    sum=0
    hits=0
    for i in range(K):
        if s[i] in ids:
            hits+=1
            sum+=1.0*hits/(i+1)
    if hits==0:
        return 0
    else:
        if method:
            K=K if K<len(ids) else len(ids) # add this check, version by LibRec
            return sum/K
        return sum/hits    


def DCG(K, l, w,method=0):
    # [[iId,score,rating],...]
    s = sorted(l, key=lambda x: x[1], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]] #fixed this bug
    return np.dot(x, w[:K])

def iDCG(K, l, w,method=0):
    s = sorted(l, key=lambda x: x[2], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]]
    return np.dot(x, w)

def nDCG(K, l,method=1):
    """
    k:int
    l:list[(int ,int ,int)]
    id,pred_score,rating
    """
    if len(l)<K:
        K=len(l)
    w = [1/log2(2+i) for i in range(K)]
    dcg = DCG(K, l, w,method)
    idcg = iDCG(K, l, w,method)
    return dcg/idcg


def Precision_Recall(K,scores,right):
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]]
    m=set(s) & set(right)
    p=len(m)/K
    r=len(m)/len(right)
    return p,r


def Precision_Recall_tha(K,scores,tha):
    # (i, model.predict(u, i), r)
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    pred_tha= s[K][1] if K<len(s) else s[-1][1]-1
    labels=[1 if x[2]>tha else 0 for x in scores]
    preds=[1 if x[1]>pred_tha else 0 for x in scores]

    p=precision_score(labels,preds)
    r=recall_score(labels,preds,zero_division=0)
    return p,r



def pr_ndcg_evulation_tha(model, Ks, train_path, test_path, n_itmes,tha=4):
    """
    """
    res_d=defaultdict(list)
    df = pd.read_csv(test_path)

    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        scores = []
        for row,one in group.iterrows():
            u, i, r = [int(one["userId"]), int(one["movieId"]), one["rating"]]
            scores.append((i, model.predict(u, i), r))
        shuffle(scores)
        for K in Ks:
            p, r = Precision_Recall_tha(K, scores, tha)
            res_d['pre_{}'.format(K)].append(p)
            res_d['rec_{}'.format(K)].append(r)
            res_d['ndcg{}'.format(K)].append( nDCG(K, scores))

    mean_res={}
    for k,v in res_d.items():
        mean_res[k]=np.mean(v)
    return mean_res

