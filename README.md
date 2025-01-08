# Requirements

```bash
conda create -n cf-gsf python=3.10
conda activate cf-gsf
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
conda install numpy==1.22.3
pip install scipy==1.9.3
conda install jupyter
conda install tqdm
conda install matplotlib
```
# Hyperparameters
* $\alpha$: `pre_val`
* $\beta$: `post_val`
* $D^{\beta} R D^{\alpha}$: `power` 

# Examples
```bash
# Amazon-book (Recall@20: 0.0735, NDCG@20: 0.0607)
python main.py --default polyfilter --model pf_chebii --dataset amazon-book --order 8 --weights 1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0.0

# Gowalla (Recall@20: 0.1923, NDCG@20: 0.1612)
python main.py --default polyfilter --model pf_chebii --dataset gowalla --order 5 --weights 1.0,0.8,0.6,0.4,0.2,0.0 --ideal_num 512 --ideal_weight 0.4 --pre power --pre_val -0.5 --post power --post_val 0.5

# Gowalla (Recall@20: 0.1925, NDCG@20: 0.1600)
python main.py --default polyfilter --model pf_chebii --dataset gowalla --order 5 --weights 1.0,0.8,0.6,0.4,0.2,0.0 --ideal_num 512 --ideal_weight 0.4 --pre power --pre_val -0.4 --post power --post_val 0.4

# Yelp2018 (Recall@20: 0.0716, NDCG@20: 0.0590)
python main.py --default polyfilter --model pf_chebii --dataset yelp2018 --order 5 --weights 1.0,0.8,0.6,0.4,0.2,0.0 --ideal_num 256 --ideal_weight 0.3 --pre power --pre_val -0.1 --post power --post_val 0.1
```
