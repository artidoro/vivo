#!/bin/bash
for filename in ~/vivo/log/$1/*.pt
do
    echo "$filename"
    #/bin/bash "~/vivo/scripts/eval.sh $filename $2 $3 $4"
    python main.py\
    --checkpoint_path "$1-eval-all"\
    --src_language "$2"\
    --trg_language "$3"\
    --device "cuda:$4"\
    --load_checkpoint_path "$filename"\
    --use_checkpoint_args\
    --unk_replace\
    --eval_batch_size 128\
    --mode eval\
    -o mode\
    -o unk_replace\
    -o eos_vector_replace
    
    echo "Chckpoing for above result waas: $filename"
done

