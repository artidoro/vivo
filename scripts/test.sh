#!/bin/bash
python main.py\
    --checkpoint_path "test-$5"\
    --src_language "$2"\
    --trg_language "$3"\
    --device "cuda:$4"\
    --load_checkpoint_path "$1"\
    --use_checkpoint_args\
    --unk_replace\
    --mode test\
    --verbose\
    -o verbose\
    -o mode\
    -o unk_replace\
    -o device
