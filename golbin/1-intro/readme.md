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

### troubleshooting
#### pip를 통해서 텐서플로우가 안 깔리는 현상
```
“Could not find a version that satisfies the requirement tensorflow (from versions: ) No matching distribution found for tensorflow”
```
: python을 32비트 버전으로 설치해서 발생.. 64비트로 재설치 후 진행하면 잘 깔린다.

#### Could not find 'msvcp140.dll' 발생
Microsoft Visual C ++ Redistributable package 재설치하면 해결된다.
https://www.drivereasy.com/knowledge/how-to-fix-msvcp140-dll-is-missing/

#### No module named '_pywrap_tensorflow_internal'
위와 유사한 문제같은데 진행이 더 안되네.. 일단 리눅스에서만 해보는걸로 하자.