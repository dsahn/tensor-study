# 기본 신경망 구현

다층 신경망을 간단하게 구현해본다고 한다.

## 인공 신경망의 작동 원리

입력값 X에 가중치W를 곱하고, 편향 b를 더한 뒤 활성화 함수(sigmoid, ReLU 등)을
거쳐 결과 y를 만들어 내는 것. 원하는 y를 만들기 위해 W와 b를 변경해가며 적절한
값을 찾는것 -> 학습, 훈련

- y = Sigmoid(X * W + b) 
- 활성화 함수 : 인공신경망을 통과해혼 값을 최종적으로 어떤값으로 만들 지
  결정하며, sigmoid, relu, tanh 가 있다.
- 이 뉴런들로 입력층, 은닉층, 출력층 구성함

