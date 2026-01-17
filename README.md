<p align="center">
  <img src="https://github.com/kps333/eScreen-beta/blob/main/img/lab_logo.png" height="100" title="lab_logo">
  <img src="https://github.com/kps333/eScreen-beta/blob/main/img/eScreen_logo.png" height="100" title="project_logo">
</p>

# eScreen-beta
eScreen is a sequence-sensitive model built upon the Striped Hyena2 architecture designed to learn interpretable regulatory context model from CRISPR perturbation experiment. Using the results of CRISPR perturbation experiment analysis and information about transcriptional factor motif, eScreen learns functional regulatory syntax and predicts regulatory activity of cis-regulatory elements.
<p align="center" style="margin-bottom: 0px;">
  <img src="https://github.com/kps333/eScreen-beta/blob/main/img/Schema_0.png" width="400" title="logo">
</p>
<p align="center" style="margin-top: 0px;">
  <img src="https://github.com/kps333/eScreen-beta/blob/main/img/Schema_1.png" width="400" title="logo">
</p>

This repository contains the official implementation of the model described in our paper:<br>Decoding the functional regulatory syntax at single-nucleotide resolution through deep learning and genome-scale perturbation.
For more details read our manuscript or access our [web site](https://escreen.huanglabxmu.com).

## Table of Contents
- [eScreen](#eScreen-beta)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Detail Demo](#Demo)
  - [Data](#data)
  - [Installation](#installation)
  - [Model Architecture](#model-architecture)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

## Quick Start
### Load demo dataset
```python
import escreen
import pickle,json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

with open('../data/celltype.dict','rb') as file:
    cell_type_dict = json.load(file)

with open('../data/demo_dataset.pkl','rb') as file:
    Demo_Dataset = pickle.load(file)
    
trainset = Demo_Dataset['Trainset'].reset_index(drop=True)
testset  = Demo_Dataset['Testset'].reset_index(drop=True)
validset = Demo_Dataset['Validset'].reset_index(drop=True)

trainset['one hot'] = None
for i,row in tqdm(trainset.iterrows(),total=len(trainset)):
    trainset.at[i,'one hot'] = escreen.genome_tool.one_hot(row['sequence'])
testset['one hot'] = None
for i,row in tqdm(testset.iterrows(),total=len(testset)):
    testset.at[i,'one hot']  = escreen.genome_tool.one_hot(row['sequence'])
validset['one hot'] = None
for i,row in tqdm(validset.iterrows(),total=len(validset)):
    validset.at[i,'one hot'] = escreen.genome_tool.one_hot(row['sequence'])

train_ds = escreen.SequenceDataset(trainset, cell_type_dict, z_col_name='cell_line')
test_ds  = escreen.SequenceDataset(testset , cell_type_dict, z_col_name='cell_line')
valid_ds = escreen.SequenceDataset(validset, cell_type_dict, z_col_name='cell_line')

train_loader = DataLoader(train_ds , batch_size=32)
test_loader  = DataLoader(test_ds  , batch_size=32)
valid_loader = DataLoader(valid_ds , batch_size=32)
```
### Train
```python
import torch

motifs_f, motifs_r, motif_names, motif_length = escreen.motif_tool.load_pwm_from_meme_c(
    "../data/Vierstra637motifs.meme", max_length=35
)
seed = 114514
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
kernel_fwd  = torch.tensor(motifs_f,dtype=torch.float)
kernel_rev  = torch.tensor(motifs_r,dtype=torch.float)
d_in = None
d_model = 512
num_filters = 512

model = escreen.eScreen(
    kernel_fwd = kernel_fwd,
    kernel_rev = kernel_rev,
    d_model=d_model,
    num_filters=num_filters,
    seq_length=500,
    celltype_num=32,
    lr=1e-5,
    device='cuda',
)

torch.cuda.empty_cache()
model.fit(train_loader,valid_loader=test_loader,epochs=50,lr=1e-4,check_step=500,earlystop=10,device='cuda',save_name='./eScreen_model')
```

### Prediction
```python
motifs_f, motifs_r, motif_names, motif_length = escreen.motif_tool.load_pwm_from_meme_c(
    "../data/Vierstra637motifs.meme", max_length=35
)
seed = 114514
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
kernel_fwd  = torch.tensor(motifs_f,dtype=torch.float)
kernel_rev  = torch.tensor(motifs_r,dtype=torch.float)
d_in = None
d_model = 512
num_filters = 512

model = escreen.eScreen(
    kernel_fwd = kernel_fwd,kernel_rev = kernel_rev,d_model=d_model,
    num_filters=num_filters,seq_length=500,celltype_num=32,lr=1e-5,device='cuda',
)

model.load_state_dict( torch.load('./eScreen_model.best.pt',map_location='cuda') )
p,y = model.predict(valid_loader,device='cuda',verbose=True,with_true=True)
``` 

## Demo
| Name | Description |
|-----------------|-------------|
|[Demo.ipynb](https://github.com/kps333/eScreen-beta/blob/main/Tutorial/Demo.ipynb)|A detailed tutorial on how to Train `eScreen` and use it to predict the activity of regulatory elements|
|[Analysis.ipynb](https://github.com/kps333/eScreen-beta/blob/main/Analysis/Analysis.ipynb)|A demo on how to reproduce some results in our paper|

## Data
All demo used data can be gain in this repository. Model weights is available at Google Drive：xxxx

## Setup
### Requirements
We recommend using our packaged setup script to create suitable environment:
```bash
git clone https://github.com/kps333/eScreen-beta.git
cd eScreen-beta
bash setup.sh
conda activate eScreen
```
### Installation
Then, you can install eScreen with pip:
```bash
pip install -e .
```

## Model Architecture

eScreen is a sequence-sensitive model built upon the Striped Hyena2 architecture, integrating:  

  ☛ Short- and long-range convolution layers for multi-scale regulatory feature extraction  

  ☛ An optional graph neural network (GNN) module that incorporates epigenetic context  

<p align="center">
  <img src="https://github.com/kps333/eScreen-beta/blob/main/img/Schema_2.png" width="600" title="logo">
</p>

## License

This project is licensed under the MIT License.

## Citation

If you use eScreen in your research, please cite our paper:

Decoding the functional regulatory syntax at single-nucleotide resolution through deep learning and genome-scale perturbation

## Contact

For questions or support, please open an issue or contact us. Please don't hesitate to contact us if you have any questions or suggestions about eScreen:
<br>[21620241153548@stu.xmu.edu.cn](mailto:21620241153548@stu.xmu.edu.cn).
<br>[sluo112211@163.com](mailto:sluo112211@163.com).
