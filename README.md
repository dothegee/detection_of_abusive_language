# detection_of_abusive_language
## 문제 정의
- 기계적으로 잡기 힘든 욕설 탐지
  - chat-gpt에서 "십1할"과 같은 한글로 응용된 단어를 이해하지 못함
## 가설
  1. 변형된 욕설 탐지 -> 단어(토큰)이 아닌 자모음의 위치조합의 Convolution 활용
  2. 맥락에 따른 혐오 표현 -> 트랜스 포머
  로 개선할 수 있지 않을까?

## 모델 선정
1. NLP기반이므로 transformer 기반 모델
2. 비속어 탐지는 실시간으로 활용 -> 속도가 빠른 트랜스포머 모델 찾기
3. 실제 활용되기 위해서는 빠르지만 가벼운 모델이여야함
4. 후보군
  > 1) MobileBERT
  > 2) TinyBERT
  > 3) FastBERT
5. 가벼우면서 빠른 장점을 가진 MobileBERT 선정

## MobileBERT 특징
1. Quantization
2. Knowledge Distillation
