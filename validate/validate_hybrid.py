from gensim.models import FastText
import util.utils as util
from Levenshtein import distance as levenshtein


# FastText 모델을 불러오는 함수
def load_fasttext_model(model_path):
    model = FastText.load(model_path)
    return model


# 유사도 기반으로 예측하는 함수
def predict_with_similarity(model, input_name, threshold_similarity=0.95):
    if input_name in model.wv.key_to_index:  # 정확하게 일치하는 경우
        return input_name  # 원래 성분명을 그대로 반환

    similar_words = model.wv.most_similar(input_name, topn=5)

    # 유사도가 90% 이상인 경우 반환
    for similar_word, similarity in similar_words:
        if similarity >= threshold_similarity:
            return similar_word

    return None  # 유사도가 90% 미만인 경우 None 반환


# 편집 거리 계산 함수
def levenshtein_distance(word1, word2):
    """편집 거리 계산 함수"""
    return levenshtein(word1, word2)


# 하이브리드 교정 함수
def hybrid_correction(input_word, model, threshold_similarity=0.97):
    predicted_output = predict_with_similarity(model, input_word, threshold_similarity)

    # 유사도 확인 및 교정 결정
    if predicted_output:
        return predicted_output

    # 유사도가 97% 미만인 경우, 편집 거리로 확인
    all_words = model.wv.index_to_key  # 모델의 모든 단어 가져오기
    closest_word = min(all_words, key=lambda word: levenshtein_distance(input_word, word))
    edit_distance = levenshtein_distance(input_word, closest_word)  # 편집 거리 계산

    input_length = len(input_word)

    # 길이에 따른 편집 거리 임계값 설정
    if input_length <= 10:
        edit_distance_threshold = 2
    else:
        edit_distance_threshold = 5

    # 편집 거리가 임계값 이하일 경우 원래 단어를 반환
    if edit_distance > edit_distance_threshold:
        return input_word  # 원래 단어 유지
    else:
        return closest_word  # 편집 거리가 가까운 단어 반환


# 모델 평가 함수
def evaluate_hybrid_model(test_file, model):
    correct_predictions = 0
    total_entries = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    for entry in test_data:
        entry = entry.strip()  # 줄바꿈 문자 제거
        if entry:
            input_name, expected_output = entry.split()  # 실제 포맷에 맞게 조정 필요

            # 자모 단위로 변환
            jamo_input = util.jamo_sentence(input_name)
            predicted_output = hybrid_correction(jamo_input, model)  # 하이브리드 교정 결과 가져오기

            # 예측 결과를 단어로 변환
            final_prediction = util.jamo_to_word(predicted_output)

            # 예측 결과와 기대 출력을 비교
            if final_prediction == expected_output:
                correct_predictions += 1
            else:
                print(f"입력: {input_name}, 기대 출력: {expected_output}, 예측: {final_prediction}")

            total_entries += 1

    # 정확도 계산
    accuracy = (correct_predictions / total_entries) * 100 if total_entries > 0 else 0
    print(f"하이브리드 교정 정확도: {accuracy:.2f}% ({correct_predictions}/{total_entries})")


# 실행 예시
model_path = '../model/fasttext'  # FastText 모델 경로
test_file = 'corpus_mecab_validate.txt'  # 검증 데이터 파일

# FastText 모델 불러오기
fasttext_model = load_fasttext_model(model_path)

# 모델 평가 수행
evaluate_hybrid_model(test_file, fasttext_model)
