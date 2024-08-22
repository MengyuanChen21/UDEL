# Uncertainty-aware Dual-evidential Learning for Weakly-supervised Temporal Action Localization

### Environment
- Python 3.10
- PyTorch == 1.13.1

### Setup
```shell script
# create & activate a conda virtual environment
conda create -n udel python=3.10
conda activate udel

# install pytorch and cudatoolkit
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# install other python libs.
pip install wandb tensorboard pandas joblib geomloss
```

## Data Preparation

**THUMOS-14 dataset:**

We use the 2048-d features provided by MM 2021 paper: Cross-modal Consensus Network for Weakly Supervised Temporal Action Localization. You can get access of the dataset from [Google Drive](https://drive.google.com/file/d/1SFEsQNLsG8vgBbqx056L9fjA4TzVZQEu/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1nspCSpzgwh5AHpSBPPibrQ?pwd=2dej). The annotations are included within this package.

## Inference
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1FUS6uBPtUxYCglkP4FKDPMbQE6ZkpRja/view?usp=sharing), and put them into "./ckpt/", and run:
```
./test.sh
```
