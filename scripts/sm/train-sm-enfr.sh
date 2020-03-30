python main.py\
    --checkpoint_path sm_deen\
    --device cuda:0\
    --src_language en\
    --trg_language fr\
    --enc_bidirectional\
    --input_feed\
    --tie_embed\
    --unk_replace\
    --fasttext_embeds_path ~/data/corpus.fasttext.fr\
    --eval_epochs 1\
#    --enc_num_layers 2\