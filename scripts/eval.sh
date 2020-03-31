#!/bin/bash
python main.py\
    --checkpoint_path "vmf-$1$2-eval-all"\
    --src_language "$1"\
    --trg_language "$2"\
    --device "cuda:$3"\
    --load_checkpoint_path "$path$filename"\
    --use_checkpoint_args\
    --unk_replace\
    --mode eval\
    -o mode\
    -o unk_replace