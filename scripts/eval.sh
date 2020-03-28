python main.py --checkpoint_path sm\
               --src_language de\
               --trg_language en\
               --enc_bidirectional\
               --input_feed\
               --tie_embed\
               --unk_replace\
               --mode eval\
               --load_checkpoint_path sm/28-03-2020_12:58:25.pt\
               --device cuda:5
