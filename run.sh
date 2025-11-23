#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=29557
#'--wavcaps',
 #'--audiocaps
 #'--clotho',
python -m d25_t6.train --enable_intra_modal_alignment --enable_alignment_loss --data_path=data --batch_size 32 --exp_name=alignment_clotho --clotho
