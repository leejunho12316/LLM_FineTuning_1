

# Dataset
<img src="https://camo.githubusercontent.com/e70f2a6a8c8f5bf0f4211dd32a0b5311c7464b65098006e654986f6738bfe034/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7261772f6d61696e2f68756767696e67666163655f6875622e737667">

RAW Data Link : https://huggingface.co/datasets/leejunho12316/seoul-mayor-hope <br>
Labeled Data Link : https://huggingface.co/datasets/leejunho12316/seoul-mayor-hope-labeled-backup2500


huggingface 데이터셋 README 도 작성

## 1. Raw Data
서울시 응답소 민원 Q&A 데이터셋
**서울시 응답소 - 시장에게 알린다**
https://eungdapso.seoul.go.kr/req/mayor_hope/mayor_hope.do

서울시 응답소 공식 홈페이지에 공개된 민원 데이터를 Web Crawling하여 수집.
시민이 서울시장에게 직접 민원·건의 사항을 올리면, 서울시가 답변하는 Q&A 형태의 공개 데이터.

<img src="./readme_res/1_eungdapso_screenshot.png">

- 데이터셋 구조

| 컬럼명            | 설명              |
|----------------|-----------------|
| `title`        | 민원 제목           |
| `Date`         | 민원 접수 날짜        |
| `Question`     | 시민이 작성한 민원 내용   |
| `Answer`       | 서울시 답변 내용       |
| `rceptNo_enc`  | 암호화된 민원 고유 접수번호 |


- 데이터셋 정제

13,540행 -> 13,184행

원본 대비 **356건**의 이상 데이터(결측값, 중복, 특정 유형 등)를 제거.

- 결측값(NaN) 제거: `Question` 또는 `Answer`가 비어있는 행 삭제.
- 중복 데이터 제거: `Question` 중복 행 삭제 -> 도배글, 어그로성 글, 중복 비난글 다수.
- 기타 데이터 제거 : 첨부된 이미지나 파일이 있지만 서버/홈페이지의 문제로 유실되고 내용을 알 수 없는 데이터. 기다 다른 목적으로 입력한 데이터. 길이가 비정상적으로 짧은 데이터 등.

## 2. Labeled Data

민원 데이터에 대한 키워드 분류 label을 생성한 Data. 실제 현장이라면 공무원이 직접 분류했을 카테고리들을 LLM을 사용해 생성.

**System Prompt**<br>
서울시 조직도를 참고하여 부서 별 맡은 역할을 System Prompt로 작성.<br>
서울특별시 조직도 : https://org.seoul.go.kr/mobile/org/orgChart.do

**with_structured_output** <br>
BaseModel을 상속받는 사용자 정의 데이터형식 클래스와 with_structured_output을 사용하여 JSON 형식으로 일관된 출력 제한.

**shufle**
원본 dataset을 shuffle해 특정 기간 데이터에 한정되지 않도록

```
SYSTEM_PROMPT = """당신은 서울시 민원 분류 담당관입니다. 지금부터 민원과 민원에 대한 답변을 읽고 키워드를 추출해주세요.
민원은 제목인 Title과 본문인 Question으로 구분되어 입력됩니다.
민원에 대한 답변은 Answer로 입력됩니다.

1. importance
Title과 Question을 보고 해당 민원의 중요도를 파악해 높음, 보통, 낮음 중 레이블을 구분하세요.
- 높음 : 행정적 조치, 전문적인 도움이 필요한 글. 특정한 문제가 발생했거나 부당한 처우에 대한 항의.
- 보통 : 보통의 의견이나 제안, 생각을 담은 글. 소식, 칭찬, 정보를 담은 글 등.
- 낮음 : 감정적으로만 작성한 글. 어그로성 글. 특정 개인에 대한 근거 없고 맹목적인 비난 글. 비논리적이고 문맥에 일관성이 없는 글. 작성이 온전히 다 되지 않은 글. 등

2. department
다음은 서울시의 각 부서가 담당하는 분야입니다. 민원 내용을 보고 해당 민원이 전달되어야 할 부서를 골라주세요.

- 교통실 : 버스·지하철·택시, 대중교통 정책, 자전거·킥보드·보행, 주차, 신호, 불법주정차, 한강버스, 교통카드, 도로교통, 자율주행
- 복지실 : 기초생활보장, 저소득층 지원, 노숙인, 어르신 돌봄, 장애인 지원, 아동·청소년 복지, 한부모·다문화가족, 중장년 지원
- 경제실 : 창업·스타트업, 소상공인·전통시장 지원, 청년 취업·일자리, 중소기업 자금, 소비자 권익, 생활임금·노동정책, 자영업자, 지원금
- 기후환경본부 : 쓰레기·재활용, 소각장, 미세먼지·대기질, 동물보호, 탄소중립·신재생에너지, 친환경차·전기차 충전, 도시공원, 식품안전
- 문화본부 : 도서관, 박물관·문화시설, 공연·예술 지원, 문화유산, 전통문화, 관광 계획, 공원 시설 관리/조성
- 시민건강국 : 보건소, 응급의료, 감염병·방역, 정신건강, 예방접종, 치매 예방, 공중위생, 건강증진, 마약 대응, 금연 지원, 금연구역 관리
- 재난안전실 : 재난대응, 취약시설 점검, 도로·보도 안전, 대피소, 시민안전보험, 제설, 인파 안전관리, 도로공사 안전 관리, 공사현장 관리
- 주택실 : 재개발·재건축, 공공주택, 전세사기, 건축인허가, 도시계획, 주거환경개선, 도시재생, 시설물 관리, 공공시설 관리, 부동산, 사유지
- 여성가족실 : 보육·어린이집, 저출생 대응, 아동학대 예방, 청소년 지원·보호, 성폭력·성희롱 예방, 디지털성범죄, 여성 안전, 양성평등
- 분류 보류 : 정부 부서 관할 이외의 기관에 대한 내용. 정치적인 내용.

단, Answer를 제외한 민원(Title과 Question)을 보았을 때 다음의 경우에 해당한다면 '분류 보류'를 설정하세요.
- Title과 Question만으로 민원의 주제를 알 수 없어 특정 부서를 분류할 수 없는 경우
- 첨부 파일을 업로드 했다고 되어 있으나 Title과 Question만으로 어떤 내용인지 유추할 수 없는 경우.
- Title과 Question이 내용을 알 수 없을 정도로 짧은 경우.

3. complaint_type
Title과 Question을 보고 민원의 유형을 다음 중 하나로 구분하세요
- 신고 : 불법 행위, 위험 상황, 규정 위반 등 제3자나 시설에 대한 문제를 알리는 경우
- 문의 : 제도, 정책, 절차, 방법 등에 대한 정보나 안내를 요청하는 경우
- 건의 : 정책 개선, 시설 설치, 제도 변경 등을 제안하는 경우
- 항의 : 행정 처리나 처우에 대한 불만을 표출하거나 시정을 요구하는 경우
- 칭찬 : 공무원, 서비스, 정책 등에 대한 긍정적인 평가를 담은 경우.
- 그 외 : 위 유형 중 어느 것으로도 분류되지 않는 경우.

4. emotion
Title과 Question을 보고 민원인의 감정상태를 긍정, 중립, 부정 중 하나로 구분하세요.

"""

```


**비용 & 정확도**

비용과 label 정확도를 고려하여 label 하기 위해 비용과 label 정확도를 각각 실험.
[2.ModelSelection](./2.ModelSelection)에 각 모델별 테스트용 label 생성 데이터, 정답 데이터 있음.

비용 : 50건의 데이터 labeling 후 처리 가격과 특이사항 분석<br>
정확도 : 수동으로 50건의 민원에 대한 정답 데이터셋 생성 후 정답률 도출.




| 모델명                      | 50건 처리 가격 (달러) | 특이사항                                       |
  |--------------------------|---------------|--------------------------------------------|
  | gpt-4o-mini              |  <0.01 (10원 미만) | 비용 최저                                      |
  | gpt-4o                   | 0.27 (400원)  | TPM 자주 걸려 ERROR 다수 발생                      |
  | claude-sonnet-4-20250514 | 0.6 (890원)     | 레거시 모델. 같은 가격에 훨씬 높은 성능을 가진 sonnet 4.6이 있음 |
  | claude-sonnet-4-6        | 0.6 (890원)     | 처리 5분 넘게 걸림                                |
  | claude-haiku-4-5-20251001 | 0.2 (300원)     | 없음                                         |
  | gemini-3-flash-preview  | (354원)         | 처리 5분 넘게 걸림.                               |

<img src="./2.ModelSelection/model_evaluation.png">

평가
1. gpt-4o-mini
- 가장 중요한 '전달 부서'의 정답률이 높음. 하지만 중요도에 대한 이해도가 **'굉장히'** 아쉬움
- 비용 아주 쌈.

2. gpt-4o
- 레이블에 대한 이해도 준수.
- 비용 준수.
- 하지만 TPM (Token Per Minute) 제한에 많이 걸려 굉장히 불편함.

3. claude-sonet-4
- 비용과 성능 대비 전달 부서 레이블에 대한 이해가 아쉬움.
- 비용 제일 비쌈.

4. **claude-haiku-4-5-20251001**
- 가장 중요한 '전달 부서'의 정답률이 높음
- 중요도, 민원 유형에 대한 판단률 준수.
- 가격은 가장 싼 gpt-4o-mini와 claude-sonnet의 중간 정도.

5. gemini-3-flash-preview
- 중요도, 전달부서 레이블에 대한 이해도가 떨어짐. 
- 비용이 저렴하지도 않음.

-> **claude-haiku-4-5-20251001**

예산 10,000원인 관계로 현재는 2500개 label 데이터 제작해 사용.


# Fine Tuning 과정

system prompt 리스트 shuffle -> 일반화 삭제

# Fine Tuning 성과

모델 : Qwen/Qwen2.5-0.5B-Instruct

### Training Loss
![training_loss.png](3.Fine_Tuning/training_loss.png)
trainer_state에 전체 loss 경과 저장 -> 1450번째 checkpoint 사용

### FineTuned & Base Model 출력 비교

Test Dataset 중 민원 데이터 10개 넣고 출력 비교

Base Model 출력
```
response:분류 보류
--------------------------------------------------
response:분류 보류
감정적으로 작성한 글. 특정 개인에 대한 근거 없고 맹목적인 비난 글. 비논리적이고 문맥에 일관성이 없는 글. 작성이 온전히 다 되지 않은 글. 등
--------------------------------------------------
response:분류 보류
--------------------------------------------------
response:분류 보류
complaint_type: 신고
emotion: 부정
--------------------------------------------------
response:분류 보류
--------------------------------------------------
response:분류 보류
complaint_type: 건의
emotion: 부정
--------------------------------------------------
response:5
6
7
8
9
10
11
12
13
14
15
16
...
--------------------------------------------------
response:분류 보류
부서: 주택실
--------------------------------------------------
response:분류 보류
--------------------------------------------------
response:importance: 높음
department: 교통실
complaint_type: 건의
emotion: 긍정
--------------------------------------------------
```

Fine-Tuned 출력

```
    response:
{"importance": "높음", "department": "경제실", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "교통실", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "경제실", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "주택실", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "여성가족실", "complaint_type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "문화본부", "complaint_type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "경제실", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "낮음", "department": "분류 보류", "complaint_Type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "문화본부", "complaint_type": "항의", "emotion": "부정"}
==================================================
    response:
{"importance": "높음", "department": "여성가족실", "complaint_Type": "건의", "emotion": "부정"}
==================================================
```

| Base Model | Fine Tuned Model|
| --- | --- | 
| - System Prompt의 지시사항 이해 불가. <br>- JSON 형식의 출력 불가능. <br> - 항목에 대한 설명을 같이 출력하거나 4가지 키워드를 전체 출력하지 못함. | - JSON 형식을 지키며 구조화된 출력이 가능.


[Fine-tuned-checkpoint1450] 파싱 실패: 0건 <br>
  importance: 367/505 = 72.7%<br>
  department: 353/505 = 69.9%<br>
  complaint_type: 356/505 = 70.5%<br>
  emotion: 415/505 = 82.2%<br>

[Base] 파싱 실패: 500건<br>
  importance: 0/0 = 0.0%<br>
  department: 0/0 = 0.0%<br>
  complaint_type: 0/0 = 0.0%<br>
  emotion: 0/0 = 0.0%<br>

Fine-Tuned Model Test 정답표

![accuracy_stacked_plot.png](3.Fine_Tuning/accuracy_stacked_plot.png)



# 추가
 full_results_checkpoints.json (checkpoint 별 label 정확도 평가) 시각화