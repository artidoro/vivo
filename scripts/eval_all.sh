#!/bin/bash
for filename in ~/vivo/log/$1/*.pt
do
    echo "$filename"
    #/bin/bash "~/vivo/scripts/eval.sh $filename $2 $3 $4"
    python main.py\
    --checkpoint_path "vmf-$2$3-eval-all"\
    --src_language "$2"\
    --trg_language "$3"\
    --device "cuda:$4"\
    --load_checkpoint_path "$filename"\
    --use_checkpoint_args\
    --unk_replace\
    --mode eval\
    -o mode\
    -o unk_replace
    
    echo "Chckpoing for above result waas: $filename"
done

