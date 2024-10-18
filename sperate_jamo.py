import util.utils as util
from tqdm import tqdm


def process_jamo(tokenized_corpus_fname, output_fname):
    toatal_lines = sum(1 for line in open(tokenized_corpus_fname, 'r', encoding='utf-8'))

    with open(tokenized_corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:

        for _, line in tqdm(enumerate(f1), total=toatal_lines):
            sentence = line.replace('\n', '').strip()
            processed_sentence = util.jamo_sentence(sentence)
            f2.writelines(processed_sentence + '\n')

tokenized_corpus_fname = 'corpus_mecab.txt'
output_fname = 'corpus_mecab_jamo.txt'
process_jamo(tokenized_corpus_fname, output_fname)