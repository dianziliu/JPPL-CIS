<h1 align="center">
A Joint Pairwise and Pointwise Learning Method based on Controversial Items Sampling
</h1>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="">Paper</a> •
  <a href="#citation">Citation</a>
</p>



Repo for submation of KDD'25: [A Joint Pairwise and Pointwise Learning Method based on Controversial Items Sampling]

<h2 id="introduction">1. Introduction✨</h2>

Recommender systems help users find more interesting items by mining their historical behaviors to understand their preferences. The training loss of the recommendation models evolves from pointwise loss to pairwise loss. The pairwise loss is well adapted to the inherent problems in recommender systems, such as missing true negative samples, and brings good performance. However, since the goal of the pairwise loss is to maximize the score difference between positive and negative samples without limitation, the positive samples become extremely large while the negative samples become extremely small. The system enhances the experience of some users while hurting the experience of others. In this paper, we propose a rating-based joint learning method (RJL), which explores the possibility of jointly learning pairwise and pointwise losses based on explicitly rating data. Specifically, RJL contains a novel loss function to find the overlapping solution spaces of the two loss functions by simultaneously optimizing the pairwise and pointwise losses. The overlapping solution space is more rigorous and has a higher probability of finding an optimal solution because it eliminates suboptimal solutions through mutual interaction. Additionally, we develop a novel controversial items sampling method, which extracts negative samples from rating data, avoiding potential systematic errors from hypothetical labels of unobserved items introduced by the pointwise loss. Experimental results on four datasets show the proposed method's superior performance compared to state-of-the-art methods, with an average increase of 5.12% on NDCG@5.


<h2 id="quick-start">2. Quick Start🚀</h2>

1. File Structure

```
.\data [training data]  
.\src  
```

2. Compile source files

```sh
g++ -std=c++11 -o src/RJL.run src/RJL.cpp 
```


3. training the model

```sh
python src/run.py
```




<h2 id="citation">3. Citation☕️</h2>

If you find this repository helpful, please consider citing our paper when it has been accepted.

<!-- ```
@inproceedings{liu-ickm-2023-tse,
  author       = {Zhen Yang and
                  Junrui Liu and
                  Tong Li and
                  Di Wu and
                  Shiqiu Yang and
                  Huan Liu},
  editor       = {Ingo Frommholz and
                  Frank Hopfgartner and
                  Mark Lee and
                  Michael Oakes and
                  Mounia Lalmas and
                  Min Zhang and
                  Rodrygo L. T. Santos},
  title        = {A Two-tier Shared Embedding Method for Review-based Recommender Systems},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {2928--2938},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3614770},
  doi          = {10.1145/3583780.3614770},
  timestamp    = {Wed, 22 Nov 2023 13:37:55 +0100},
}
``` -->
