import os
import sys
sys.path.append(".")
from time import sleep
from datetime import datetime
import numpy as np

from model import Simi

from process_data import process_data_by_split

from multiprocessing import Pool

from utils.evulation import pr_ndcg_evulation_tha

import pandas as pd


from collections import defaultdict

paths = [
         "data/Ml100Krating.csv",
         "data/ML1Mratings.csv",
         "data/YahooR4.csv",
         "data/ML10Mratings.csv",]

                      

# ML100K,ML1M,YahooR3,YahooR4
n_userss=[6400,6400,8000,80000]
n_itmess=[4000,4000,12000,80000]

dataset=["ml100k","ml1m","r4","ml10m"]

dim=10
lr=0.007
rg=0.05
iter=30


sample_methods={
    'CIS':process_data_by_split,

}

def evulation_all_datasets_by_datasplit(sample_type,repeat=5):    
    method_name="JPPL_CIS"
    dir = "res/res_{}".format(sample_type)
    
    if not os.path.exists(dir):
        os.mkdir(dir)

    
    sample_function=sample_methods[sample_type]
    
    Ks1= [1,2,5,10]
    # Ks1=[1,2,5,10,50,100]
    Ks2=[10,20]
    experiment     = os.path.join(dir,"res2.txt")
    train_path_fmt = os.path.join(dir,"{}/train{}.csv")
    test_path_fmt  = os.path.join(dir,"{}/test{}.csv")
    u_path         = os.path.join(dir,"{}/u{}.bin")
    i_path         = os.path.join(dir,"{}/i{}.bin")

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # traversing each datset
            # evulation for each dataset
            if name not in ['ml10m']:
                continue
            res_d=defaultdict(list)
            if not os.path.exists(os.path.join(dir,name)):
                os.mkdir(os.path.join(dir,name))

            print("dataset",name)
            print("dataset",name,file=f,flush=True)  

            
            # 多进行求数据集
            # print("spliting data....")
            # dataSplitPool=Pool(10)
            for i in range(repeat):
                train_path = train_path_fmt.format(name,i)
                test_path  = test_path_fmt.format(name,i)
                if not (os.path.exists(train_path) and os.path.exists(test_path)):
                    # dataSplitPool.apply_async(sample_function,(path,",",n_items,train_path,test_path))
                    sample_function(path,',',n_items,train_path,test_path,0.1,0.1)
            # dataSplitPool.close()
            # dataSplitPool.join()

            for i in range(repeat):
                train_path=train_path_fmt.format(name,i)
                test_path=test_path_fmt.format(name,i)
                
                upath=u_path.format(name,i)
                ipath=i_path.format(name,i)
                exe="src_JPPR/RJSPR.run"

                cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                -train_path {} -test_path {} -u_path {} -i_path {}  -a {} -b {}".format(
                    exe,n_users,n_items,dim,iter,lr,rg,
                    train_path,test_path,upath,ipath,0.2,0.2
                )
                if not os.path.exists(upath):                
                    os.system(cmd)
                    print("training over!")
                model=Simi(method_name,n_users,n_items,dim,train_path)
                model.load(upath,ipath)
                model.name=method_name
                #
                # this_res_d= pr_ndcg_evulation_Alls_mt(model,Ks1,train_path,test_path,n_items)
                this_res_d= pr_ndcg_evulation_tha(model,Ks1,train_path,test_path,n_items)
                for k,v in this_res_d.items():
                    res_d[k].append(v)
                    
                print(
                    '\t'.join(["{}:{}".format(k,np.mean(v)) for k,v in res_d.items()])
                )
                print(
                    '\t'.join(["{}:{:.6f}".format(k,np.mean(v)) for k,v in res_d.items()]),file=f,flush=True
                )
                # break



if __name__ == "__main__":
    for sampler in sample_methods.keys():
        print('start to run sampler: {}'.format(sampler))
        evulation_all_datasets_by_datasplit(sampler,repeat=1)
        # break