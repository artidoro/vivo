#!/bin/bash
for filename in $1"/*.pt;" do
    python main.py --checkpoint_path eval_all\
               --src_language $2\
               --trg_language $3\
               --device cuda:$4\
               --load_checkpoint_path $path$filename\
               --use_checkpoint_args\
               --unk_replace\
               --mode eval\
               -o mode\
               -o unk_replace
    echo "Results for checkpoint:\n"$path$filename
done