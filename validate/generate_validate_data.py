import random


def generate_ocr_error(name):
    ocr_error = {
        "바": "파", "비": "니", "타": "다", "카": "가",
        "아": "야", "프": "뜨", "디": "미", "핀": "민",
    }
    result = list(name)
    for i in range(len(result)):
        k = 0
        if result[i] in ocr_error:
            if k == 2:
                break
            result[i] = ocr_error[result[i]]
            k += 1
    return ''.join(result)


def add_certification(name):
    certs = ['(KP)', '(USP)', '(KPC)', '(KHP)', '(JP)']
    return f"{name}{random.choice(certs)}"


def process_chemical_names(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        names = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, name in enumerate(names):
            name = name.strip()  # 줄바꿈 제거
            if name:
                if i % 2 == 0:  # 짝수 인덱스일 때는 OCR 오류 생성
                    ocr_error_name = generate_ocr_error(name)
                    f.write(f"{ocr_error_name}")
                else:  # 홀수 인덱스일 때는 인증 표기 추가
                    certified_name = add_certification(name)
                    f.write(f"{certified_name}")

                f.write(f" {name}\n")



# 실행 예시
input_file = '../train/corpus_mecab.txt'  # 원본 성분명 목록 파일
output_file = 'corpus_mecab_validate.txt'  # 변환된 결과 파일

process_chemical_names(input_file, output_file)