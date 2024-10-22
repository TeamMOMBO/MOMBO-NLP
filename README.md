# MOMBO-NLP

## 1. 프로젝트 개요


이 프로젝트는 임부금기 의약품 성분의 정확한 인식과 분류를 목표로 하고 있다. [한국의약품안전관리원의 임부금기 성분 데이터](https://www.drugsafe.or.kr/iwt/ds/ko/useinfo/EgovDurInfoSerPn.do)를 활용하여 성분명 데이터베이스를 구축하고, OCR 오류 및 실제 데이터를 고려한 성능 검증을 수행한다. 이를 위해 **워드 임베딩** 기술을 사용하여 자연어 처리 모델을 개발하였고, 성분명의 효과적인 인식과 분류를 위해 **편집 거리 알고리즘**을 통한 하이브리드 교정 방법을 제시한다.


## 2. 모델링 방법
### 2.1. 말뭉치 구성 및 전처리
한국어 FastText 모델링에 사용한 말뭉치는 [한국의약품안전관리원의 임부금기 성분 데이터](https://www.drugsafe.or.kr/iwt/ds/ko/useinfo/EgovDurInfoSerPn.do)를 이용하여 구성하였다. 수집된 파일에서 국문 성분명은 [약학정보원](https://www.health.kr/main.asp) 기준으로 부여되었으며, 문장 부호, 수식, 특수 문자 등은 제거하여 전처리를 수행 하였다. 정제된 말뭉치는 총 1199개의 토큰으로 구성되어 있다.

### 2.2. 자모 분리 학습 방법
한국어의 음절 구조인 초성, 중성, 종성을 반영하기 위해 자모 분리 방식을 적용했다. 각 한글 음절을 분리하여 **FastText** 모델에 학습시킴으로써, 한국어의 특수한 음 구조를 모델이 더욱 잘 이해하도록 하였다.

> 예시) 프로게스테론 -> ㅍㅡ-ㄹㅗ-ㄱㅔ-ㅅㅡ-ㅌㅔ-ㄹㅗㄴ

이를 통해 실제 데이터 혹은 OCR 인식 오류 발생 시에도 성분명을 보다 효과적으로 인식하고 예측할 수 있다.

## 3. 검증 방식 및 오류 수정
### 3.1. [검증 데이터 세트](validate/corpus_mecab_validate.txt) 구성
모델 검증을 위해 다음 두 가지 패턴으로 변형된 성분명을 구성했다.
1. OCR 오류 단어 생성 (600개)
   - 각 성분명에 대해 0~2개의 문자 오류를 적용
2. 인증 방식 추가 (600개)
   - 성분명 뒤에 (KP), (USP), (KPC) 와 같은 인증 표기를 추가
3. 교정이 필요 없는 데이터 (100개)
   - 일반적인 텍스트 

> 인증 방식은 [의약품 표시 등에 관한 규정](https://www.mfds.go.kr/brd/m_211/view.do?seq=14459&srchFr=&srchTo=&srchWord=%EC%9D%98%EC%95%BD%ED%92%88%EC%9D%98+%ED%92%88%EB%AA%A9%EF%BF%BD&srchTp=&itm_seq_1=0&itm_seq_2=0&multi_itm_seq=0&company_cd=&company_nm=&page=26)에 따라 성분명 뒤에 인증 방식이 오는 경우를 반영 했다. 이를 통해 모델이 실제 데이터에서의 형식을 정확하게 인식하는지 판단 했다.

### 3.2. [기존 검증 방식](validate/validate.py)의 오류
실제 모델을 배포해본 결과, 기존 검증 방식은 유사도가 높은 단어를 무조건 교정하는 방식이 문제로 지적되었다. 이로인해 일반적인 단어들도 유해 성분으로 불필요하게 교정되는 현상이 발생했고 이러한 문제를 해결하기위해 교정을 위한 유사도 임계치를 90% 이상으로 설정하였다. 이로 인해 전체적인 교정율이 크게 하락하였다. 특히 짧은 단어에서는 교정율이 현저히 낮아져, 검증 데이터에 대한 교정률이 30% 이상으로 떨어졌다.

## 4. [하이브리드 교정](validate/validate_hybrid.py) 방식
이러한 문제를 해결하기 위해 **FastText 모델의 유사도 기반 예측**과 **편집 거리(Levenshtein Distance)** 알고리즘을 결합한 방식으로, 교정이 필요한 경우와 아닌 경우를 보다 정확하게 구분했다.

### 4.1. 하이브리드 교정 수식
```math
\hat{w} = 
\begin{cases} 
w_{sim} & \text{if} \ s(w, w_{sim}) \geq \tau_s \\
w_{edit} & \text{if} \ d(w, w_{edit}) \leq \tau_d \ \text{and} \ s(w, w_{sim}) < \tau_s \\
w & \text{otherwise}
\end{cases}
```

### 4.2. 수식 설명

- $\( \hat{w} \)$: 최종 교정된 단어
- $\( w \)$: 입력 단어
- $\( w_{sim} \)$: FastText 모델에서 예측한 가장 유사한 단어
- $\( s(w, w_{sim}) \)$: 입력 단어와 예측 단어 간의 유사도
- $\( \tau_s \)$: 유사도 임계값 (본 프로젝트에서는 97%로 설정)
- $\( w_{edit} \)$: 편집 거리 기준으로 가장 가까운 단어
- $\( d(w, w_{edit}) \)$: 입력 단어와 편집 거리 기반으로 교정된 단어 간의 편집 거리
- $\( \tau_d \)$: 편집 거리 임계값 (본 프로젝트에서는 10자 이하의 경우 2, 그 이상은 5로 설정)

### 3. 하이브리드 교정 로직

1. 입력 단어 $\( w \)$가 FastText 모델 내에 존재하면, 해당 단어를 그대로 반환한다.
2. FastText 모델에서 가장 유사한 단어 $\( w_{sim} \)$의 유사도가 임계값 $\( \tau_s \)$ 이상일 경우, 해당 단어를 교정 단어로 사용한다.
3. 유사도가 임계값 미만인 경우, 편집 거리 $\( d(w, w_{edit}) \)$를 계산해 편집 거리가 임계값 $\( \tau_d \)$ 이하인 경우에만 교정한다.
4. 그 외의 경우에는 원본 단어를 반환한다.

## 성능 검증 결과

새로운 하이브리드 교정 방식을 적용한 결과, 정확도는 크게 향상되었다.

### 5.1 최종 정확도

- **하이브리드 교정 방식의 정확도**: **99.85%** (1300개의 테스트 샘플 중 1298개에서 올바른 예측을 수행)

기존 방식에서 문제가 되었던 불필요한 교정을 최소화하였으며, 짧은 성분명에서도 높은 정확도를 보였다.

