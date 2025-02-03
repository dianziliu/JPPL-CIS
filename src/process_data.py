
from random import choices,choice,randint

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import numpy as np

def sigmod(x):
    return 1/(1+np.exp(-x))

def process_data_by_split(path,sep,n_items,train_path,test_path,alpha,bete,n_ng=4,test_size=0.5):
    
    """
    训练文件以四元组的形式保存 "u,i,j,q"  
    测试集以原始文件格式保存.csv
    """
    df=pd.read_csv(path,sep=sep,engine="python")

    train_file=open(train_path,"w")
    # test_file=open("test_file.csv","w")
    
    df2=pd.DataFrame()

    train_parts=[]
   
    # 1. 划分数据集
    print("划分数据集")
    for uid,group in df.groupby(["userId"]):
        
        train,test=train_test_split(group,test_size=test_size)
        train_parts.append([uid,train])
        df2=pd.concat([df2,test])
    df2.to_csv(test_path,index=False)

    # 2. 统计概率
    print("统计概率")
    item_metrix=np.zeros([n_items,n_items])  # 记录两个物品之间关系的数量
    item_metrix_max=np.zeros([n_items,n_items])  # 记录i>j的数量
    for uid, train in train_parts:
        # 对用户的评分进行统计
        temp={}
        for rating, r_group in train.groupby("rating"):
            temp[int(rating)]=list(r_group.movieId.unique())
        
        u_ratings=sorted(temp.keys(),reverse=True)
        for idx,v in enumerate(u_ratings[:-1]):
            i_items=temp[v]
            for i in i_items:
                num=0
                
                # 相等关系
                for temp_i in i_items:
                    item_metrix[i,temp_i]+=1
                    item_metrix[temp_i,i]+=1
                # 大于关系
                for vj in u_ratings[idx+1:]:
                    j_list=temp[vj]
                    num+=len(j_list)
                    for j in j_list:
                        item_metrix_max[i,j]+=1
                        item_metrix[i,j]+=1
                        item_metrix[j,i]+=1
    item_metrix_p=np.divide(item_metrix_max,item_metrix)
    item_metrix_p[np.isnan(item_metrix_p)]=0
    
    # 3. 
    print("3: 准备训练集")
    for uid, train in train_parts:
        temp={}
        for rating, r_group in train.groupby("rating"):
            temp[int(rating)]=list(r_group.movieId.unique())

        # 得到用户的历史物品
        max_rating=max(temp.keys())
        min_rating=min(temp.keys())
        u_ratings=sorted(temp.keys(),reverse=True)
        # for i in range(max_rating,min_rating,-1):/
        for idx,v in enumerate(u_ratings[:-1]):
            i_items=temp[v]
            ng_rating=temp[u_ratings[idx+1]]
            
            for i in i_items:
                # 相似采样
                weights=[]
                for pos in i_items:
                    if pos ==i:
                        weights.append(0)
                        continue
                    p1=item_metrix_p[i,pos]
                    p3=item_metrix_p[pos,i]
                    p2=1-p1-p3
                    if p1+p3==0:
                        weights.append(1+bete)
                    else:
                        weights.append(1+bete*sigmod(p2/(p1+p3)))
                positems=choices(i_items,weights,k=n_ng)
                
                
                weights2=[]
                for neg in ng_rating:
                    p1=item_metrix_p[i,neg]
                    p3=item_metrix_p[neg,i]
                    p2=1-p1-p3
                    if p2==0:
                        weights2.append(1+alpha)
                    else:
                        weights2.append(1+alpha*sigmod((p2+p3)/p1))
                negitems=choices(ng_rating,weights2,k=n_ng)
                    
                for j,k in zip(positems,negitems):
                    train_file.write("{},{},{},{}\n".format(
                        uid, i,j,k))
    train_file.close()