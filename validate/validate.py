from gensim.models import FastText
import util.utils as util


# FastText 모델을 불러오는 함수
def load_fasttext_model(model_path):
    model = FastText.load(model_path)
    return model


# 모델 예측을 시뮬레이션하는 함수
def model_predict(model, input_name):
    # FastText를 사용하여 입력 이름에 대한 예측을 수행
    if input_name in model.wv.key_to_index:  # 정확하게 일치하는 경우
        return input_name  # 원래 성분명을 그대로 반환

    predicted_label, _ = model.wv.most_similar(input_name)[0]
    return predicted_label  # 가장 높은 확률의 레이블 반환


# 모델 평가 함수
def evaluate_model(test_file, model):
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
            predicted_output = model_predict(model, jamo_input)  # 모델 예측 결과 가져오기

            # 예측 결과를 단어로 변환
            final_prediction = util.jamo_to_word(predicted_output)

            # 예측 결과와 기대 출력을 비교
            if final_prediction == expected_output:
                correct_predictions += 1
            else:
                print(final_prediction, expected_output)
            total_entries += 1

    # 정확도 계산
    accuracy = (correct_predictions / total_entries) * 100 if total_entries > 0 else 0
    print(f"모델 정확도: {accuracy:.2f}% ({correct_predictions}/{total_entries})")


# 실행 예시
model_path = '../model/fasttext'  # FastText 모델 경로
test_file = 'corpus_mecab_validate.txt'  # 검증 데이터 파일

# FastText 모델 불러오기
fasttext_model = load_fasttext_model(model_path)

# 모델 평가 수행
evaluate_model(test_file, fasttext_model)