# Learning 3D Anisotropic Noise Distributions Improves Molecular Force Fields

This repository contains the official PyTorch implementation of the work "Learning 3D Anisotropic Noise Distributions Improves Molecular Force Fields" .


The code is modified and implemented based on the [DeNS](https://github.com/atomicarchitects/equiformer) repository code. In this repository, we provide code implementations for PCQM4Mv2 pre-training and MD17 fine-tuning. Implementations for other datasets can be done by referring to similar modifications.


## Environment Setup


### Environment 
Run the following command to automatically install the environment:
```
conda env create -f env/env_equiformer.yml
conda activate AniDS
cd ocp/fairchem/
pip install -e .
cd ../..
```

### PCQM4Mv2

The dataset of PCQM4Mv2 will be automatically downloaded when running training.


### MD17

The dataset of MD17 will be automatically downloaded when running training.


## Training


### PCQM4Mv2

1. We can train AniDS by running:

    ```bash
        python train.py --conf ./config/PCQ/PCQM4Mv2-4A100.yaml --job-id pretraining --test_type AniDS
    ```

2. The PCQM4Mv2 dataset will be downloaded automatically as we run training for the first time.

3. Model weights will be saved under the `log_dir` path specified in `config/PCQ/PCQM4Mv2-4A100.yaml`.



### MD17

1. We provide training scripts under [`scripts/train/md17/equiformer/equiformer_AniDS/finetune`](scripts/train/md17/equiformer_AniDS/finetune).
For example, we can train Equiformer for the molecule of `aspirin` by running:

    ```bash
        sh ./scripts/train/md17/equiformer_AniDS/finetune/target@aspirin.sh  
    ```

2. Finetune logs of Equiformer can be found [`md17_logs`](md17_logs) ($L_{max} = 2$). Note that the units of energy and force are kcal mol $^{-1}$ and kcal mol $^{-1}$ Å $^{-1}$.

## Acknowledgement

Our implementation is based on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [e3nn](https://github.com/e3nn/e3nn), [timm](https://github.com/huggingface/pytorch-image-models), [ocp](https://github.com/Open-Catalyst-Project/ocp), [SEGNN](https://github.com/RobDHess/Steerable-E3-GNN), [TorchMD-NET](https://github.com/torchmd/torchmd-net), []() and [DeNS](https://github.com/atomicarchitects/equiformer).

## Citation
Please consider citing the works below if this repository is helpful:
```
@inproceedings{liu2025learning,
  title={Learning 3d anisotropic noise distributions improves molecular force fields},
  author={Liu, Xixian and Jiao, Rui and Liu, Zhiyuan and Liu, Yurou and Liu, Yang and Lu, Ziheng and Huang, Wenbing and Zhang, Yang and Cao, Yixin},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
