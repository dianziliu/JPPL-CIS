import os
import sys
sys.path.append(".")
from time import sleep
from datetime import datetime
import numpy as np

from model import Simi
from process_data import process_data_by_split
from until.evulation import (ndcg_evaluations, ndcg_map_evulations,pred_to_save,
                             pr_ndcg_evulation_Alls)

import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

paths = ["data/MLLSratings.csv",
         "data/ML10Mratings.csv",
         "data/Ml100Krating.csv",
         "data/ML1Mratings.csv",
         "data/YahooR3.csv",
         "data/YahooR4.csv"]

                      

# ML100K,ML1M,YahooR3,YahooR4
n_userss=[800,80000,6400,6400,16000,8000]
n_itmess=[10000,80000,4000,4000,1024,12000]

dataset=["mlls","ml10m","ml100k","ml1m","r3","r4"]

dim=10
lr=0.007
rg=0.05
iter=30


def evulation_all_datasets(repeat=5):    
    method_name="PPR_JSPR"
    experiment="src_JPPR/res2/res.txt"
    Ks1=[5,10]
    Ks2=[10,20]
    a,b=(0.2,0.2)
    dir="src_JPPR/res2/{}"
    train_path_fmt = dir+"/train{}.csv"
    test_path_fmt = dir+"/test{}.csv"
    u_path = dir+"/u{}.bin"
    i_path = dir+"/i{}.bin"

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["ml1m"]:
            #     continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = {5: [], 10: []}
            All_maps = {10: [], 20: []} 
            if not os.path.exists(dir.format(name)):
                os.mkdir(dir.format(name))

            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            dataSplitPool=Pool(10)
            for i in range(repeat):
                if name in ["ml10m"]:
                    break
                train_path=train_path_fmt.format(name,i)
                test_path=test_path_fmt.format(name,i)
                if not (os.path.exists(train_path) and os.path.exists(test_path)):
                    dataSplitPool.apply_async(process_data_by_split,(path,",",n_items,train_path,test_path,1,1))
            dataSplitPool.close()
            dataSplitPool.join()            
            
            for i in range(repeat):
                train_path=train_path_fmt.format(name,i)
                test_path=test_path_fmt.format(name,i)
                if not (os.path.exists(train_path) and os.path.exists(test_path)):
                    process_data_by_split(path,",",n_items,train_path,test_path,1,1)
                
                upath=u_path.format(name,i)
                ipath=i_path.format(name,i)
                exe="src/RJL.run"

                cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                -train_path {} -test_path {} -u_path {} -i_path {} -a {} -b {}".format(
                    exe,n_users,n_items,dim,iter,lr,rg,
                    train_path,test_path,upath,ipath,a,b
                )
                print("training over!")
                os.system(cmd)
                model=Simi(method_name,n_users,n_items,dim,train_path)
                model.load(upath,ipath)
                model.name=method_name
                ndcgs=ndcg_evaluations(model,Ks1,test_path)
                for k in Ks1:
                    All_ndcgs[k].append(np.mean(ndcgs[k]))

            print("method {},\nndcg{}:{},ndcg{}:{}".format(
                method_name, 5, np.mean(
                                All_ndcgs[5]), 10, np.mean(All_ndcgs[10]),), file=f, flush=True)
            #     ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
            #     for k in Ks1:
            #         All_ndcgs[k].append(np.mean(ndcgs[k]))
            #     for k in Ks2:
            #         All_maps[k].append(np.mean(maps[k]))

            # print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
            #     method_name, 5, np.mean(
            #                     All_ndcgs[5]), 10, np.mean(All_ndcgs[10]),
            #                     10, np.mean(All_maps[10]), 20, np.mean(All_maps[20])
            #     ), file=f, flush=True)
 

def evulation_all_datasets_iters():    
    print("evulation_all_datasets_iters")
    method_name = "PPR"
    experiment  = "src_PPR/iter_exp/res.txt"
    Ks1=[5,10]
    Ks2=[10,20]
    dir            = "src_PPR/temp_data/{}"
    train_path_fmt = "src_PPR/temp_data/{}/train{}.csv"
    test_path_fmt  = "src_PPR/temp_data/{}/test{}.csv"
    u_path         = "src_PPR/temp_data/{}/iter_u{}.bin"
    i_path         = "src_PPR/temp_data/{}/iter_i{}.bin"

    df={"dataset":[],"iter":[],"ndcg5":[],"ndcg10":[],"map10":[],"map20":[] }

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            
            for local_iter in range(1,30):
                # traversing each datset
                # evulation for each dataset
                All_ndcgs = {5: [], 10: []}
                All_maps = {10: [], 20: []} 
                if not os.path.exists(dir.format(name)):
                    os.mkdir(dir.format(name))

                print("dataset",name)
                print("dataset",name,file=f,flush=True)  

                for i in range(5):
                    train_path=train_path_fmt.format(name,i)
                    test_path=test_path_fmt.format(name,i)
                    if not (os.path.exists(train_path) and os.path.exists(test_path)):
                        raise "error"
                        process_data_by_split(path,",",n_items,train_path,test_path)
                    
                    upath=u_path.format(name,i)
                    ipath=i_path.format(name,i)
                    exe="src_PPR/PPR.run"

                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {}".format(
                        exe,n_users,n_items,dim,local_iter,lr,rg,
                        train_path,test_path,upath,ipath
                    )
                    print("training over!")
                    os.system(cmd)
                    model=Simi(method_name,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                    for k in Ks1:
                        All_ndcgs[k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[k].append(np.mean(maps[k]))

                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    method_name, 5, np.mean(
                                    All_ndcgs[5]), 10, np.mean(All_ndcgs[10]),
                                    10, np.mean(All_maps[10]), 20, np.mean(All_maps[20])
                    ), file=f, flush=True)
    

                # 记录实验结果
                df["dataset"].append(name)
                df['iter'].append(local_iter)
                df['ndcg5'].append(np.mean(All_ndcgs[5]))
                df['ndcg10'].append(np.mean(All_ndcgs[10]))
                df['map10'].append(np.mean(All_maps[10]))
                df['map20'].append(np.mean(All_maps[20]))

    df2=pd.DataFrame(df)
    df2.to_csv("src_PPR/iter_exp",index=False)


def evulation_all_datasets_ngs():
    

    method_name="PPR"
    experiment="src_PPR/ng_datas/res2.txt"
    Ks1=[5,10]
    Ks2=[10,20]
    dir="src_PPR/ng_datas/{}"
    train_path_fmt = "src_PPR/ng_datas/{}/train_{}_{}.csv"
    test_path_fmt = "src_PPR/ng_datas/{}/test_{}_{}.csv"
    u_path = "src_PPR/ng_datas/{}/u_{}_{}.bin"
    i_path = "src_PPR/ng_datas/{}/i_{}_{}.bin"

    ngs=[1,2,4,6,8,10]

    # 准备数据集
    
    # process_datasets_mlp(dir, train_path_fmt, test_path_fmt, ngs)   

    df={"dataset":[],"ng":[],"ndcg5":[],"ndcg10":[],"map10":[],"map20":[] }

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            for ng in ngs:
                f.write("ng:"+str(ng)+"\n")
                All_ndcgs = {5: [], 10: []}
                All_maps = {10: [], 20: []} 
                if not os.path.exists(dir.format(name)):
                    os.mkdir(dir.format(name))
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  

                for i in range(5):
                    train_path=train_path_fmt.format(name,i,ng)
                    test_path=test_path_fmt.format(name,i,ng)
                    if not (os.path.exists(train_path) and os.path.exists(test_path)):
                        raise "data error"
                        process_data_by_split(path,",",n_items,train_path,test_path)
                    
                    upath=u_path.format(name,i,ng)
                    ipath=i_path.format(name,i,ng)
                    exe="src_PPR/PPR.run"

                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {}".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path,test_path,upath,ipath
                    )
                    print("training over!")
                    os.system(cmd)
                    model=Simi(method_name,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                    for k in Ks1:
                        All_ndcgs[k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[k].append(np.mean(maps[k]))

                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    method_name, 5, np.mean(
                                    All_ndcgs[5]), 10, np.mean(All_ndcgs[10]),
                                    10, np.mean(All_maps[10]), 20, np.mean(All_maps[20])
                    ), file=f, flush=True)

                # 记录实验结果
                df["dataset"].append(name)
                df['ng'].append(ng)
                df['ndcg5'].append(np.mean(All_ndcgs[5]))
                df['ndcg10'].append(np.mean(All_ndcgs[10]))
                df['map10'].append(np.mean(All_maps[10]))
                df['map20'].append(np.mean(All_maps[20]))

    df2=pd.DataFrame(df)
    df2.to_csv("src_PPR/ng_datas/res_by_pd.txt",index=False)

def process_datasets_mlp(dir, train_path_fmt, test_path_fmt, ngs):
    train_paths = []
    test_paths  = []
    p_Paths     = []
    p_n_items   = []
    all_ngs     = []

    # 多进行处理数据集
    for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
        if not os.path.exists(dir.format(name)):
            os.mkdir(dir.format(name))
        for i in range(5):
            for ng in ngs:
                train_path=train_path_fmt.format(name,i,ng)
                test_path=test_path_fmt.format(name,i,ng)  
                #
                p_Paths.append(path)
                train_paths.append(train_path)
                test_paths.append(test_path)
                p_n_items.append(n_items)
                all_ngs.append(ng)
    print("start to split data!")
    with ProcessPoolExecutor(10) as executor:
        executor.map(
            process_data_by_split,p_Paths,[","]*len(p_Paths),p_n_items,train_paths,test_paths,all_ngs
        )

if __name__ == "__main__":
    evulation_all_datasets()
    # evulation_all_datasets_ngs()
    # evulation_all_datasets_iters()
