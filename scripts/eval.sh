#!/bin/bash
python main_all.py\
    --checkpoint_path "vmf-$2$3-eval-all"\
    --src_language "$2"\
    --trg_language "$3"\
    --device "cuda:$4"\
    --load_checkpoint_path "$1"\
    --use_checkpoint_args\
    --unk_replace\
    --mode eval\
    -o mode\
    -o unk_replace
