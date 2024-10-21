from gensim.models import FastText
from tqdm import tqdm
import logging

corpus_fname = 'corpus_mecab_jamo.txt'
model_fname = 'model/fasttext'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('corpus 생성')
corpus = [sent.strip().split(" ") for sent in tqdm(open(corpus_fname, 'r', encoding='utf-8').readlines())]

print("학습 중")
model = FastText(corpus, vector_size =100, workers=4, sg=1, epochs=200, min_count=1)
model.save(model_fname)

print(f"학습 소요 시간 : {model.total_train_time}")
# https://projector.tensorflow.org/ 에서 시각화 하기 위해 따로 저장
model.wv.save_word2vec_format(model_fname + "_vis")
print('완료')