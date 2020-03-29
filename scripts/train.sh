python main.py --checkpoint_path vmf\
               --loss_function vmf\
               --device cuda\
               --src_language de\
               --trg_language en\
               --enc_bidirectional\
               --input_feed\
               --tie_embed\
               --unk_replace\
               --fasttext_embeds_path ~/data/corpus.fasttext.txt
