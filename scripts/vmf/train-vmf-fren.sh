python main.py \
    --checkpoint_path vmf-fren \
    --device cuda \
    --dropout 0.0 \
    --enc_bidirectional \
#    --enc_num_layers 2\
    --fasttext_embeds_path ~/data/corpus.fasttext.txt \
    --input_feed \
    --loss_function vmf \
    --lr 5e-4 \
    --max_len 100 \
    --mode train \
    --src_language fr \
    --train_epochs 20 \
    --eval_epochs 1 \
    --trg_language en \
    --unk_replace \
    --vmf_lambda_1 0.02 \
    --vmf_lambda_2 0.1 \
    --write_to_file \
