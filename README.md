# Fine-Tuning LLM
**개요**

이 프로젝트는 LoRA(Low-Rank Adaptation) 기법으로 모델 레이어에 학습 가능한 Low-Rank 행렬을 삽입해, 적은 파라미터로도 모델의 일반화 능력을 유지하며 Fine-Tuning하는 PEFT (Parameter Efficient Fine Tuning) 프로젝트입니다. Fine-Tuning 된 모델은 HuggingFace에 업로드하여 누구나 사용할 수 있도록 했습니다.

**목적**

Fine-Tuning의 목적은 천만 서울시민의 의견을 접수받는 온라인 민원신청 창구 '서울시 응답소'에 업로드 되는 수 많은 민원 글들을 인력의 개입 없이 그 중요도와 책임 소재가 따르는 부서를 분석해 빠르게 분류하여 민원 처리의 효율성 증대와 시간 단축을 달성하기 위함입니다. 이를 위해 서울시 응답소에 업로드 되어 있는 공개된 민원 글들을 Web Crawling을 통해 수집하고 Fine-Tuning용 데이터세트로 정제하여 훈련에 사용했습니다. 


# Dataset
<img src="https://camo.githubusercontent.com/e70f2a6a8c8f5bf0f4211dd32a0b5311c7464b65098006e654986f6738bfe034/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7261772f6d61696e2f68756767696e67666163655f6875622e737667">

RAW Data Link : https://huggingface.co/datasets/leejunho12316/seoul-mayor-hope <br>
Labeled Data Link : https://huggingface.co/datasets/leejunho12316/seoul-mayor-hope-labeled-backup2500


huggingface 데이터셋 README 도 작성

## 1. Raw Data
서울시 응답소 민원 Q&A 데이터셋
**서울시 응답소 - 시장에게 알린다**
https://eungdapso.seoul.go.kr/req/mayor_hope/mayor_hope.do

2011년 4분기부터 2026년 4월 중순까지 서울시 응답소 공식 홈페이지에 공개된 민원 데이터를 Web Crawling하여 수집.
시민이 서울시장에게 직접 민원·건의 사항을 올리면, 서울시가 답변하는 Q&A 형태의 공개 데이터.

<img src="./readme_res/1_eungdapso_screenshot.png">

- 데이터셋 예시

| 분류   | `title`<br>민원 제목                   | `Date`<br>민원 접수 날짜 | `Question`<br>민원 내용                                       | `Answer`<br>답변 내용                                            |`rceptNo_enc`<br>암호화된 민원 고유 접수번호
|------|------------------------------------|--------------------|-----------------------------------------------------------|--------------------------------------------------------------|-----------------|
| 1    | 서울의 공공디자인은 문화컨텐츠 스토리텔링으로 리모델링을---- | 2011-12-31         | 시장님! 서울의 디자인서울정책은 공공디자인 정책으로 많은 변화를 가져왔습니다. 물론 부정적 이미지... | 000 님 안녕하십니까? 서울시장 박원순입니다. 서울의 공공디자인을 문화컨텐츠 스토리텔링으로 리모델링... | X-cCuvM...
| 5000 | 우이동 경전철 공사로인한 도로및 지하실 침하           | 2015-08-11         |"2012년 부터 시작된 건물앞 경전철 공사로 인한 건물의 균열 도로침하,지하실 물난리등 수차래 ...|"000님 안녕하세요?\r\n000님의 메일 잘 받아보았습니다. 우이-신설 경전철공사로 인해 도로 침하 및 지하실 균열로 건물에...|qwhcvXJ2gq...
| 9999 | 목동123단지 도계위 통과 요청                  | 2018-05-02         |"시장님,\r\n이제 목동123단지 3종 환원을 담은 목동지구단위 계획이 다음주면 주민공람을...|"시정발전에 깊은 관심을 가지고 협조하여 주심에 감사의 말씀을 드립니다.\r\n \n목동아파..| iC_fJLG9...
| ...  | ...                                | ...                |...|...|...




- 데이터셋 정제

13,540행 -> 13,184행

원본 대비 **356건**의 이상 데이터(결측값, 중복, 특정 유형 등)를 제거.

- 결측값(NaN) 제거: `Question` 또는 `Answer`가 비어있는 행 삭제.
- 중복 데이터 제거: `Question` 중복 행 삭제 -> 도배글, 어그로성 글, 중복 비난글 다수.
- 기타 데이터 제거 : 첨부된 이미지나 파일이 있지만 서버/홈페이지의 문제로 유실되고 내용을 알 수 없는 데이터. 기다 다른 목적으로 입력한 데이터. 길이가 비정상적으로 짧은 데이터 등.

## 2. Labeled Data

민원을 분석해 LLM이 최종적으로 출력해야 할 label을 생성.
실제 현장이라면 공무원의 도움을 받아 직접 분류했을 작업을 LLM을 사용해 진행.

- 데이터셋 예시

|Raw Data와 동일|assistant|
|---|---|
|...|"{""importance"": ""높음"", ""department"": ""경제실"", ""complaint_Type"": ""항의"", ""emotion"": ""부정""}"
|...|{"importance": "높음", "department": "주택실", "complaint_Type": "항의", "emotion": "부정"}
|...|{"importance": "보통", "department": "문화본부", "complaint_Type": "건의", "emotion": "부정"}
|

### System Prompt
RAGAS Prompt를 응용하여 importance, department, complaint_type, emotion 4가지의 키워드를 도출하도록 System Prompt 작성. 

**with_structured_output** <br>
BaseModel을 상속받는 사용자 정의 데이터형식 클래스와 with_structured_output을 사용하여 JSON 형식으로 일관된 출력 제한.

| # | 키워드 | 설명 |
|---|--------|------|
| 1 | `importance` | 민원의 중요도를 구분하는 label. 민원이 빠르게 처리되어 도움을 받아야 하면 높음, 일반적인 의견 전달이라면 보통, 감정적이고 비난을 담은 글이라면 낮음. |
| 2 | `department` | 해당 민원이 전달되어야 하는 부서를 판별하는 label. 서울시 조직도를 참고하여 부서 별 맡은 역할을 요약해 작성. ([서울특별시 조직도](https://org.seoul.go.kr/mobile/org/orgChart.do)) |
| 3 | `complaint_type` | 민원의 유형을 구분하는 label. 신고, 문의, 건의, 항의, 칭찬 그리고 그 외로 분류. |
| 4 | `emotion` | 민원인의 감정상태를 구분하는 label. 긍정, 중립, 부정으로 분류. |



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


### 비용 & 정확도 측정

label 생성 시 사용할 LLM 선정을 위해 비용과 정확도를 측정.

일관성 있는 labeling을 위해 비용과 정확도를 각각 실험.<br>
[2.ModelSelection](./2.ModelSelection)에 각 모델별 테스트 데이터 셋 50건에 대한 label 생성 데이터와 수동으로 제작한 labeling 정답 데이터가 있음.

<br>

1. 비용 : 50건의 데이터 처리 후 처리 가격과 특이사항 분석<br>

| 모델명                      | 50건 처리 가격 (달러) | 특이사항                                       |
  |--------------------------|---------------|--------------------------------------------|
  | gpt-4o-mini              |  <0.01 (10원 미만) | 비용 최저                                      |
  | gpt-4o                   | 0.27 (400원)  | TPM 자주 걸려 ERROR 다수 발생                      |
  | claude-sonnet-4-20250514 | 0.6 (890원)     | 레거시 모델. 같은 가격에 훨씬 높은 성능을 가진 sonnet 4.6이 있음 |
  | claude-sonnet-4-6        | 0.6 (890원)     | 처리 5분 넘게 걸림                                |
  | claude-haiku-4-5-20251001 | 0.2 (300원)     | 없음                                         |
  | gemini-3-flash-preview  | (354원)         | 처리 5분 넘게 걸림.                               |

<br>

2. 정확도 : 수동으로 50건의 민원에 대한 정답 데이터셋 생성 후 모델 별 키워드 별 정답률 도출.
<img src="./2.ModelSelection/model_evaluation.png">


| 모델 | 중요도 | 전달부서 | 민원유형 | 감정상태 |
|------|--------|----------|----------|----------|
| claude-haiku-4-5-20251001 | ➖ 38 | ✅ 42 | ✅ 39 | ➖ 38 |
| claude-sonnet-4-20250514 | ✅ 39 | ➖ 37 | ✅ 38 | ➖ 38 |
| claude-sonnet-4-6 | ✅ 40 | ❌ 34 | ➖ 37 | ✅ 42 |
| gemini-3-flash-preview | ❌ 25 | ❌ 36 | ➖ 37 | ✅ 41 |
| gpt-4o-mini | ❌ 27 | ➖ 40 | ➖ 37 | ➖ 38 |
| gpt-4o | ➖ 36 | ✅ 41 | ❌ 34 | ✅ 41 |
✅ : 준수 (상위 2등) <br>
➖ : 보통 <br>
❌ : 아쉬움 (하위 2등) <br>
-> 가장 중요한 label인 전달부서를 잘 분류하면서 '준수'항목이 2개 이상, 비용 효율적인 **claude-haiku-4-5-20251001**로 결정


예산 10,000원인 관계로 현재는 2500개 label 데이터 제작해 사용.

<br><br><br>

---

<br><br><br>

# Fine Tuning

[4_Fine_Tuning_Code_RUNPOD_ver(A40).ipynb](4_Fine_Tuning_Code_RUNPOD_ver%28A40%29.ipynb)

## 과정

### 1. 데이터셋 분할

Labeled Data를 Train : Test = 4 : 1로 분할하여 사용.

| 분할 | 개수 |
|------|------|
| Train | 2,000 |
| Test | 500 |

<br>

### 2. System Prompt - 항목별 순서 Shuffle

System Prompt 내 `importance`, `department`, `complaint_type`, `emotion` 각 항목의 레이블 리스트를 매 샘플마다 `random.shuffle()`로 순서를 섞어 데이터를 구성.

모델이 특정 레이블의 등장 순서를 암기하지 않고, 각 레이블의 **의미**를 학습하도록 유도하기 위함.

```python
def get_system_prompt():
    importance_items = [("높음", "..."), ("보통", "..."), ("낮음", "...")]
    random.shuffle(importance_items)  # 순서 섞기

    department_items = [("교통실", "..."), ("복지실", "..."), ...]
    random.shuffle(department_items)
    ...
```

<br>

### 3. Base Model

[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

- 파라미터 수 : 0.5B
- dtype : `bfloat16`
- `padding_side = "left"` — 학습 후 추론 시 배치 처리를 위해 설정

<br>

### 4. LoRA 설정

| 파라미터 | 값 | 설명 |
|---|---|---|
| `r` | 8 | LoRA 랭크. 저차원 공간의 크기. 작을수록 메모리 효율적이나 학습 능력 제한 |
| `lora_alpha` | 32 | LoRA 스케일링 계수. 가중치 업데이트가 모델에 미치는 영향 조정 |
| `lora_dropout` | 0.1 | 드롭아웃 확률. 과적합 방지 |
| `target_modules` | `["q_proj", "v_proj"]` | LoRA를 적용할 레이어. Self-Attention의 Query·Value 프로젝션에만 적용 |
| `bias` | `"none"` | 편향(bias)은 LoRA로 조정하지 않음 |
| `task_type` | `"CAUSAL_LM"` | Causal Language Modeling (시퀀스 생성) 태스크 |

<br>

### 5. SFTConfig 설정

| 파라미터 | 값 | 설명 |
|---|---|---|
| `num_train_epochs` | 3 | 전체 데이터셋 학습 반복 횟수 |
| `per_device_train_batch_size` | 2 | GPU당 배치 크기 |
| `gradient_accumulation_steps` | 2 | 그래디언트 누적 스텝. 유효 배치 크기 = 2×2 = **4** |
| `learning_rate` | 1e-4 | 학습률 |
| `lr_scheduler_type` | `"constant"` | 워밍업 이후 학습률 고정 유지 |
| `warmup_ratio` | 0.03 | 전체 스텝의 3%를 학습률 선형 증가 구간으로 사용 |
| `max_grad_norm` | 0.3 | 그래디언트 클리핑 임계값. 폭발적 그래디언트 방지 |
| `max_seq_length` | 8192 | 최대 시퀀스 길이 |
| `bf16` | `True` | bfloat16 정밀도. FP32 수준 범위 + 메모리 효율 |
| `gradient_checkpointing` | `True` | 중간 활성화값 미저장 후 재계산. 메모리 절약 |
| `optim` | `"adamw_torch_fused"` | PyTorch Fused AdamW 최적화기 |
| `save_strategy` | `"steps"` / `save_steps=50` | 50 스텝마다 체크포인트 저장 |

<br>

### 6. collate_fn — 핵심 전처리 로직

배치 내 데이터를 모델 입력 형식으로 변환하는 함수. 핵심은 **Loss 마스킹** 처리.

**동작 방식**

1. 메시지를 Chat Template으로 변환 후 토큰화
2. 전체 토큰에 대해 `labels`를 `-100`으로 초기화 (Loss 계산 제외)
3. `<|im_start|>assistant` ~ `<|im_end|>` 구간을 탐색하여 해당 토큰에만 실제 token ID를 레이블로 설정
4. 배치 내 최대 길이에 맞춰 **left padding** 적용

> `-100`은 PyTorch CrossEntropyLoss에서 무시되는 값. System·User 구간의 토큰은 Loss 계산에서 제외하고, **모델이 생성해야 하는 assistant 응답 부분만 학습**하도록 유도.

**처리 예시**

```
[입력 텍스트 구조]
<|im_start|>system
당신은 서울시 민원 분류 담당관입니다...
<|im_end|>
<|im_start|>user
Title: 불법주정차 신고합니다
Question: ...
<|im_end|>
<|im_start|>assistant
{"importance": "높음", "department": "교통실", ...}
<|im_end|>

[labels 처리 결과]
[-100, -100, ...(system 구간)..., -100,   ← 전부 -100 (Loss 제외)
 -100, -100, ...(user 구간)...,   -100,   ← 전부 -100 (Loss 제외)
 1234,  567, ...(assistant 구간)..., 999] ← 실제 token ID (Loss 계산 대상)
```

<br>

### 7. 학습 환경

| 항목       | 내용                              |
|----------|---------------------------------|
| 플랫폼      | [RunPod](https://www.runpod.io/) |
| GPU      | NVIDIA A40 (48GB VRAM)          |
| 학습 소요 시간 | 약 20분                           |
| 총 step   | 1,500 (체크포인트 50 간격으로 저장)        |
 

## 성과

### FineTuned & Base Model 출력 비교

Fine Tuning 완료된 Model과 Qwen2.5-0.5B-Instruct 모델에 동일한 prompt 10건을 입력해 출력 비교.

1. Base Model 출력
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

| Base Model                                                                                                      | Fine Tuned Model|
|-----------------------------------------------------------------------------------------------------------------| --- | 
| - System Prompt의 지시사항 이해 불가. <br> (항목에 대한 설명을 같이 출력, 4가지 키워드 중 일부만 출력, 관련이 없는 출력 등.)<br>- JSON 형식의 출력 불가능. <br> | - JSON 형식을 지키며 구조화된 출력이 가능. |


### FineTuning Checkpoint - 키워드 정답률 그래프
Fine funing step 0부터 1500까지 50간격으로 저장된 checkpoint마다 test data를 사용해 키워드 별 정답률 변동을 시각화하였다. 

![4_Fine_Tuning_Accuracy_by_Checkpoint.png](3.Fine_Tuning/4_Fine_Tuning_Accuracy_by_Checkpoint.png)

![4_Fine_Tuning_Accuracy_Combined.png](3.Fine_Tuning/4_Fine_Tuning_Accuracy_Combined.png)


## 핵심 결론
department : 프로젝트 기획 단계에서는 민원을 구분할 수 있는 명확한 기준을 세울 수 있다고 생각하고 진행하였다. 하지만 직접 민원을 읽어보고 손수 분류하며 이해를 해 갈 수록 생각이 달라졌다.
민원 주제의 가장 많은 비율을 차지하는 '교통'과 법적인 자문이 가장 많은 '주택' 분야를 제외한 나머지 분야는 책임 소재를 명확히 할 수 없었다.

importance, complaint_type: 높음, 보통, 낮음 각각의 항목에 대한 좀 더 명확한 기준을 명시해주어 데이터를 생성했어야겠다는 생각이 들었다. 이 분야의 전문가라고 할 수 있는 공무원들의 도움을 받아 레이블링을 직접 하면 공통의 기준이 나오겠지만.<br>
emotion : LLM의 기본적인 한국어 이해도가 준수해 Fine-Tuning을 진행하지 않은 상황에서도 높은 정확도에서 시작하여 큰 문제가 없었다. <br>

->
Department와 Complaint Type에서 가장 높은 정답률을 보이며 Importance와 Emotion 카테고리도 충분히 학습이 진행된 **1150번째 checkpoint**를 최종 모델로 선정한다. 


# 다음에 할 것
- VLLM 올려서 실사용 진행해보기.
- 모델 HuggingFace 올리기? HuggingFace Spaces에 Gradio UI로 배포하기? 
# README
- Fine Tuning 과정 추가
- system prompt 항목별 설명 (ragas 그렇게 나눈 이유, 부서 설정한 이유)
