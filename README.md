<p align="center">
  <img src="https://github.com/xmuhuanglab/eScreen/blob/main/img/lab_logo.png" height="100" title="lab_logo">
  <img src="https://github.com/xmuhuanglab/eScreen/blob/main/img/eScreen_logo.png" height="100" title="project_logo">
</p>

# eScreen
eScreen is a sequence-sensitive model built upon the Striped Hyena2 architecture designed to learn interpretable regulatory context model from CRISPR perturbation experiment. Using the results of CRISPR perturbation experiment analysis and information about transcriptional factor motif, eScreen learns functional regulatory syntax and predicts regulatory activity of cis-regulatory elements.
<p align="center" style="margin-bottom: 0px;">
  <img src="https://github.com/xmuhuanglab/eScreen/blob/main/img/Schema_3.png" width="1000" title="logo">
</p>

This repository contains the official implementation of the model described in our paper:<br>Decoding the functional regulatory syntax at single-nucleotide resolution through deep learning and genome-scale perturbation.
For more details read our manuscript or access our [web site](https://escreen.huanglabxmu.com).

## Table of Contents
- [eScreen](#eScreen-beta)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Detail Demo](#Demo)
  - [Data](#data)
  - [Setup](#Setup)
  - [Model Architecture](#model-architecture)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

## Quick Start
### Load demo dataset
```python
import pickle

with open(f"../data/train.pkl", "rb") as f:
    trainset = pickle.load(f)
with open(f"../data/valid.pkl", "rb") as f:
    validset = pickle.load(f)
with open(f"../data/test.pkl", "rb") as f:
    testset = pickle.load(f)

# Decode compact indices back to one-hot for the model
for ds in [trainset, validset, testset]:
    ds["sequence"] = np.eye(4, dtype=np.uint8)[ds["sequence"]]
```
### Train
```python
optimizer = AdamW([
    {"params": model.output_MoE.parameters(), "lr": 1e-3},
    {"params": model.output_header.parameters(), "lr": 3e-4},
    {"params": model.cls_head.parameters(), "lr": 3e-4},
    {
        "params": [
            p for n, p in model.named_parameters()
            if "output_MoE" not in n and "output_header" not in n and "cls_head" not in n
        ],
        "lr": 1e-3,
    },
], lr=1e-3, weight_decay=0.01)

def val_score_fn(model, val_data, device):
    preds, _ = model.predict(val_data, batch_size=384, device=device, verbose=False, with_true=False)
    return m2a_combo_score(preds, val_data["y"], val_data)

save_name = "./model/model"
if not os.path.exists("./model"):
    os.makedirs("./model",exist_ok=True)

model.fit(
    trainset, val_data=validset, batch_size=256, epochs=50, optimizer=optimizer, check_step=500,
    earlystop=12, use_boost=True, t=0.45, task="reg", device=DEVICE, save_name=save_name, 
    aux_bce_lambda=0.6, aux_loss_type="bce", focal_gamma=2.0, pos_weight=30.0, label_smoothing=0.0,
     max_grad_norm=1.0, sampler="label_boost", val_score_fn=val_score_fn
)
```

### Prediction
```python
from tqdm import tqdm

# Load best checkpoint and evaluate on test set
model.load_state_dict(
    torch.load("./model/model.best.pt", map_location=DEVICE, weights_only=False),
    strict=False,
)

preds, y_true = model.predict(testset, batch_size=384, device=DEVICE, verbose=True, with_true=True)

``` 
### Vedio tutorial
How to install `eScreen` and the dependent environment:


coming soon



  
How to run the demo:


coming soon



### Time cost
On our device (CPU: Intel Xeon Silver 4310, 24C/48T; GPU: A30 24GB), environment setup and demo execution take approximately 3 hours and ~8 mins respectively.  

## Demo
| Name | Description |
|-----------------|-------------|
|[Demo.ipynb](https://github.com/xmuhuanglab/eScreen/blob/main/Tutorial/Demo.ipynb)|A detailed tutorial on how to Train `eScreen` and use it to predict the activity of regulatory elements|
|[Figure3.ipynb](https://github.com/xmuhuanglab/eScreen/blob/main/Analysis/Figure3.ipynb)|Notebook to reproduce key results of figure 3 in our paper|
|[Figure4.ipynb](https://github.com/xmuhuanglab/eScreen/blob/main/Analysis/Figure4.ipynb)|Notebook to reproduce key results of figure 4 in our paper|
|[Figure5.ipynb](https://github.com/xmuhuanglab/eScreen/blob/main/Analysis/Figure5.ipynb)|Notebook to reproduce key results of figure 5 in our paper|

## Data
All demo used data can be gain in this repository.

## Setup
### Requirements
We recommend using our packaged setup script to create suitable environment:
```bash
git clone https://github.com/xmuhuanglab/eScreen.git
cd eScreen
bash setup.sh
conda activate eScreen
```
### Installation
Then, you can install eScreen with pip:
```bash
pip install -e .
```

## Model Architecture

eScreen is a sequence-sensitive model built upon the Striped Hyena 2 architecture, integrating:  

  ☛ Short- and long-range convolution layers for multi-scale regulatory feature extraction  


<p align="center">
  <img src="https://github.com/xmuhuanglab/eScreen/blob/main/img/Schema_4.png" width="1000" title="logo">
</p>

## License

This project is licensed under the MIT License.

## Citation

If you use eScreen in your research, please cite our [paper](https://www.biorxiv.org/content/10.64898/2026.02.02.703403v1):

Luo S, Lin L, Zhang H, et al. eScreen: a deep learning framework for functionally decoding the regulatory genome at single-nucleotide resolution[J]. bioRxiv, 2026: 2026.02. 02.703403.

## Contact

For questions or support, please open an issue or contact us. Please don't hesitate to contact us if you have any questions or suggestions about eScreen:
<br>Liquan Lin: [21620241153548@stu.xmu.edu.cn](mailto:21620241153548@stu.xmu.edu.cn).
<br>Shijie Luo: [sluo112211@163.com](mailto:sluo112211@163.com).
