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
# LastFM (Recall@20: 0.1199, NDCG@20: 0.1101
python main.py --default polyfilter --model pf_chebii --dataset last-fm --order 8 --weights 1.0,0.653,0.503,0.5,0.5,0.5,0.497,0.347,0.0 --ideal_num 256 --ideal_weight 0.1 --pre power --pre_val -0.2 --post power --post_val 0.2

# Gowalla (Recall@20: 0.1941, NDCG@20: 0.1616)
python main.py --default polyfilter --model pf_chebii --dataset gowalla --order 8 --weights 1.0,0.944,0.797,0.619,0.5,0.381,0.203,0.056,0.0 --ideal_num 512 --ideal_weight 0.2 --pre power --pre_val -0.4 --post power --post_val 0.4

# Amazon-book (Recall@20: 0.0738, NDCG@20: 0.0613)
python main.py --default polyfilter --model pf_chebii --dataset amazon-book --order 8 --weights 1.0,0.727,0.516,0.5,0.5,0.5,0.484,0.273,0.0 --ideal_num 512 --ideal_weight 0.2 --pre power --pre_val -0.0 --post power --post_val 0.0
```
