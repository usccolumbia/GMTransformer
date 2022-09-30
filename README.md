# GMTransformer 
-a probablistic generative transformer for molecular design

This repository contains the code for our paper:  
[**GENERATIVE TRANSFORMER LANGUAGE MODELS FOR GENERATIVE AND TINKERING DESIGN OF MOLECULES**](put-arxiv-link-here)  
*Lai Wei, Nihang Fu, Yuqi Song, Qian Wang, and Jianjun Hu*

by <a href="http://mleg.cse.sc.edu" target="_blank">Machine Learning and Evolution Laboratory</a>, University of South Carolina.

### Datasets for training Generative Molecular Transformer (GMTransformer)

Benchmark Datasets from Molecular Sets(MOSES): [MOSES](https://github.com/molecularsets/moses)

SELFIES tokenizers from: [Selfies](https://github.com/aspuru-guzik-group/selfies)

The GMTransformer datasets including:

SMILES-atom training dataset (1,584,664 samples)

SMILES-atom validation dataset (176,075 samples)

SELFIES-atom training dataset (1,584,664 samples)

SELFIES-atom validation dataset (176,075 samples)

They can be downloaded here:

[SELFIES_data.zip](https://github.com/usccolumbia/GMTransformer/blob/main/SELFIES_data.zip)

[SMILES_data.zip](https://github.com/usccolumbia/GMTransformer/blob/main/SMILE_data.zip)

### Running environment set up

The BLM language model code we used is from [here](https://github.com/Varal7/blank_language_model), which is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. It has been tested in PyTorch 1.6.0, PyTorch Lightning 1.0.7

Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) based on your python & cuda version
```
conda create -n blm
conda activate blm
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning=1.0.7

or 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.0.7
```


### Codebase

Go to the blank language model (BLM) repository from [https://github.com/Varal7/blank_language_model/](https://github.com/Varal7/blank_language_model)

Download the repository and unzip it. Install the necessary python libraries as instructed.

And then put a GMTransformer folder inside it.

```
git clone https://github.com/Varal7/blank_language_model.git
cd blank_language_model
mkdir GMTransformer
cd GMTransformer

```

### How to train the model with GMTransformer dataset

#### Download Data
Download datasets from the above links and put it into the GMTransformer folder, then unzip it under `SMILE_data.zip` and `SELFIES_data.zip` folder.

```
wget https://github.com/usccolumbia/GMTransformer/blob/main/SELFIES_data.zip?raw=true -O SELFIES_data.zip
wget https://github.com/usccolumbia/GMTransformer/blob/main/SMILE_data.zip?raw=true -O SMILE_data.zip
unzip SELFIES_data.zip
unzip SMILES_data.zip
```

After the above, the directory should be:

```
blank_language_model
  |-GMTransformer
          ├── SMILE_data
              ├── SMILES_atom_train.txt
              ├── SMILES_atom_valid.txt
          ├── SELFIES_data
              ├── SELFIES_atom_train.txt
              ├── SELFIES_atom_valid.txt
      └── README.md
```

#### Training
An example is to train a GMTransformer model on the SMILES_atom dataset. 
```
cd blank_language_model
python train.py --train GMTransformer/SMILE_data/SMILES_atom_train.txt --valid GMTransformer/SMILE_data/SMILES_atom_valid.txt --root_dir checkpoints/SMILES/atom/ \
--vocab_size 100 --max_len 200 --model_type blm --share_emb_prj_weight
```
The training for other models is similar to SMILES_atom dataset.

#### How to generate new molecules using the trained models
For all of the following, replace `epoch\=???.ckpt` with the checkpoint saved in training.

Generate molecules using the trained SMILES_atom model.
```
python test.py --checkpoint checkpoints/SMILES/atom/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--sample 1000 --decode sample --output sample.txt
```
The output file is located at

checkpoints/SMILES/atom/lightning_logs/version_1/outputs/sample.txt

You can then convert the generated token list into SMILES file:

```
python convert2smiles.py --input checkpoints/SMILES/atom/lightning_logs/version_1/outputs/sample.txt  --output output_smiles.txt
```

for SELFIES-model, 

```
python selfiestoken2smiles.py --input checkpoints/SELFIES/atom/lightning_logs/version_1/outputs/sample.txt  --output output2_smiles.txt
```

#### How to generate new molecules using our pretrained models

Download the zipped model file from figshare [zipped model file](https://figshare.com/articles/software/GMTransformer/21256338)
put it into the GMTransformer folder, and unzip it. 

Then run the following to generate molecules using the GMTransformer-SMILES or GMTransformer-SELFIES model.
```
python test.py --checkpoint GMTransformer/models/SELFIES-model/checkpoint/blanklm-epoch=2835-val_loss=0.78.ckpt \
--sample 1000 --decode sample --output sample.txt

python test.py --checkpoint GMTransformer/models/SMILES-model/checkpoint/blanklm-epoch=2716-val_loss=0.71.ckpt \
--sample 1000 --decode sample --output sample.txt
```
After the generation, you need to use the same conversion step as above to convert the sequences into SMILES format.



### Citation

If you use our work, please cite:

```bibtex
@article{wei2022probabilistic,
  title={Probabilistic Generative Transformer Language models for Generative Design of Molecules},
  author={Wei, Lai and Fu, Nihang and Song, Yuqi and Wang, Qian and Hu, Jianjun},
  journal={arXiv preprint arXiv:2209.09406},
  year={2022}
}

``
