python train.py --train GMTransformer/SMILE_data/SMILES_atom_train.txt --valid GMTransformer/SMILE_data/SMILES_atom_valid.txt --root_dir checkpoints/SMILES/atom/ \
--vocab_size 100 --max_len 200 --model_type blm --share_emb_prj_weight
