from gensim.models import FastText
import util.utils as util


def transform(list):
    return [(util.jamo_to_word(w), r) for (w, r) in list]


# 모델을 로딩하여 가장 유사한 단어를 출력
loaded_model = FastText.load("../model/fasttext")
print(loaded_model.wv.vectors.shape)

print(transform(loaded_model.wv.most_similar(util.jamo_sentence('프로게스테톤'), topn=5)))
print(transform(loaded_model.wv.most_similar(util.jamo_sentence('레미펜타닐1mg'), topn=5)))
print(transform(loaded_model.wv.most_similar(util.jamo_sentence('알렌드론'), topn=5)))