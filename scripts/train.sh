python main.py --checkpoint_path sm_deen_small\
               --device cuda:3\
               --src_language de\
               --trg_language en\
               --enc_bidirectional\
               --input_feed\
               --tie_embed\
               --unk_replace\
               --trg_fasttext_embeds\
               --max_len 20\
               --min_freq 2