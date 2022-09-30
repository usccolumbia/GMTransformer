# GMT-molecular
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


### Dependencies

The code is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. It has been tested in PyTorch 1.6.0, PyTorch Lightning 1.0.7

Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) given your python & cuda version


### Codebase

Go to the blank language model (BLM) repository from [https://github.com/Varal7/blank_language_model/](https://github.com/Varal7/blank_language_model)
Download the repository and unzip. And then put the GMTransformer folder inside it.

### How to train the model with GMTransformer dataset

#### Download Data
Download datasets from the above link, then unzip it under `SMILES_data.zip` and `SELFIES_data.zip` folder.
After the above, the directory should be:
```
blank_language_model
  |-GMTransformer
      ├── GMTransformer_dataset
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
python train.py --train GMTransformer_dataset/SMILE_data/SMILES_atom_train.txt --valid GMTransformer_dataset/SMILE_data/SMILES_atom_valid.txt --root_dir checkpoints/SMILES/atom/ \
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

### Citation

If you use our work, please cite:

```bibtex
@article{wei2022probabilistic,
  title={Probabilistic Generative Transformer Language models for Generative Design of Molecules},
  author={Wei, Lai and Fu, Nihang and Song, Yuqi and Wang, Qian and Hu, Jianjun},
  journal={arXiv preprint arXiv:2209.09406},
  year={2022}
}
}
``
