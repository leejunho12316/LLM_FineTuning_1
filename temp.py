--------------
department : 프로젝트 기획 단계에서는 민원을 구분할 수 있는 명확한 기준을 세울 수 있다고 생각하고 진행하였다. 하지만 직접 민원을 읽어보고 손수 분류하며 이해를 해 갈 수록 생각이 달라졌다.
민원 주제의 가장 많은 비율을 차지하는 '교통'과 법적인 자문이 가장 많은 '주택' 분야를 제외한 나머지 분야는 책임 소재를 명확히 할 수 없었다.

importance, complaint_type: 높음, 보통, 낮음 각각의 항목에 대한 좀 더 명확한 기준을 명시해주어 데이터를 생성했어야겠다는 생각이 들었다. 이 분야의 전문가라고 할 수 있는 공무원들의 도움을 받아 레이블링을 직접 하면 공통의 기준이 나오겠지만.<br>
emotion : LLM의 기본적인 한국어 이해도가 준수해 Fine-Tuning을 진행하지 않은 상황에서도 높은 정확도에서 시작하여 큰 문제가 없었다. <br>

huggingface 데이터셋 README 도 작성

# 다음에 할 것
- VLLM 올려서 실사용 진행해보기.

------------------------------------------------------------------------------------------------------------------------------------------

System Prompt 수정 후 결과 모음

#Qwen2.5-0.5b
[Fine-tuned] 파싱 실패: 0건
  importance: 349/505 = 69.1%
  department: 372/505 = 73.7%
  complaint_type: 286/367 = 77.9%
  emotion: 426/505 = 84.4%

[Base] 파싱 실패: 0건
  importance: 3/30 = 10.0%
  department: 2/30 = 6.7%
  complaint_type: 11/30 = 36.7%
  emotion: 10/30 = 33.3%

#Qwen2.5-1.5b
[Fine-tuned] 파싱 실패: 0건
  importance: 396/505 = 78.4%
  department: 409/505 = 81.0%
  complaint_type: 298/360 = 82.8%
  emotion: 442/505 = 87.5%

[Base] 파싱 실패: 0건
  importance: 14/30 = 46.7%
  department: 11/30 = 36.7%
  complaint_type: 4/30 = 13.3%
  emotion: 14/30 = 46.7%

#Qwen2.5-3b
[Fine-tuned] 파싱 실패: 0건
  importance: 421/505 = 83.4%
  department: 409/505 = 81.0%
  complaint_type: 424/505 = 84.0%
  emotion: 449/505 = 88.9%

[Base] 파싱 실패: 0건
  importance: 19/30 = 63.3%
  department: 8/30 = 26.7%
  complaint_type: 19/30 = 63.3%
  emotion: 22/30 = 73.3%

#Qwen2.5-7b
[Fine-tuned] 파싱 실패: 0건
  importance: 427/505 = 84.6%
  department: 420/505 = 83.2%
  complaint_type: 420/505 = 83.2%
  emotion: 450/505 = 89.1%

[Base] 파싱 실패: 0건
  importance: 22/30 = 73.3%
  department: 17/30 = 56.7%
  complaint_type: 21/30 = 70.0%
  emotion: 24/30 = 80.0%

