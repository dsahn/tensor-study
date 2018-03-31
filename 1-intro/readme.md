# 3분 딥러닝 읽는중
## 책 앞부분 정리
- 머신러닝 : 개념
- 딥러닝으로 대표되는 인공신경망 : 머신러닝을 구현하는 기술 중 하나
- 규칙기반 인공지능과 비교(rule-based AI)
 - 길고 노란색이고 흰거면 바나나
- 머신러닝 : 바나나 사진을 주고 특징을 컴퓨터가 알아서 파악
- 인공신경망을 이용한 알렉스넷이 이미지 인식 대회에서 인식률을 매우 높임
- 빅데이터, gpu발전, 다양한 딥러닝 알고리즘 발명덕분에 발전
- 병렬 처리 능력과 역전파 등의 알고리즘을 통해 계산량을 많이 줄임

### 설치 중 에러 증상
pip를 통해서 텐서플로우가 안 깔리는 현상
```
“Could not find a version that satisfies the requirement tensorflow (from versions: ) No matching distribution found for tensorflow”
```
: python을 32비트 버전으로 설치해서 발생.. 64비트로 재설치 후 진행하면 잘 깔린다.