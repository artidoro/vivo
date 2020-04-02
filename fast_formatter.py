import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True)
    parser.add_argument("--trg_file", required=True)
    parser.add_argument("--aligned_file",required=True)
    args = parser.parse_args()
    paired_sentences = []
    trg_sentences = []
    src_sentences = []
    with open(args.src_file, encoding='utf-8') as src_file_fh:
        src_sentences = src_file_fh.readlines()
    with open(args.trg_file, encoding='utf-8') as trg_file_fh:
        trg_sentences = trg_file_fh.readlines()
    paired_sentences = [(src.strip(),tgt.strip()) for src,tgt in zip(src_sentences, trg_sentences)]
    with open(args.aligned_file, 'w+', encoding='utf-8') as out_file_fh:
        for pair in paired_sentences:
            out_file_fh.write('{} ||| {}\n'.format(pair[0],pair[1]))

