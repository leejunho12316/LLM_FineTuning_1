# %% [markdown]
# # 개요
# Runpod GPU 업로드해 실행
# %% [markdown]
# ## 0. 기본 설정
# %%
# %pip install matplotlib
# %%
# %pip install "torch==2.4.0"
# %pip install "transformers==4.45.1" "datasets==3.0.1" "accelerate==0.34.2" "trl==0.11.1" "peft==0.13.0"
# %pip install openpyxl rich
# %% [markdown]
# # 1. 데이터 불러오기
# %%
import pandas as pd
# %%
from huggingface_hub import login
HF_TOKEN = "" #read only token
login(token=HF_TOKEN)
# %%
from datasets import load_dataset

dataset = load_dataset('leejunho12316/seoul-mayor-hope-labeled-backup2500')
df_original = dataset['train'].to_pandas()
df_original = df_original[df_original['assistant'].notna()].reset_index(drop=True)

total_length = len(df_original)
print(total_length)
# %%
# Split into train and test sets
train_df = df_original[:2000].copy()
test_df = df_original[2000:].copy()
print(len(train_df), len(test_df))
# %%
train_df.head()
# %%
test_df.head()
# %%
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
# %%
import random

def get_system_prompt():
  """구성 시마다 내용물 순서 바꿔가며 일반화 학습 안하도록"""

  # 1. importance
  importance_items = [
      ("높음", "행정적 조치, 전문적인 도움이 필요한 글. 특정한 문제가 발생했거나 부당한 처우에 대한 항의."),
      ("보통", "보통의 의견이나 제안, 생각을 담은 글. 소식, 칭찬, 정보를 담은 글 등."),
      ("낮음", "감정적으로만 작성한 글. 어그로성 글. 특정 개인에 대한 근거 없고 맹목적인 비난 글. 비논리적이고 문맥에 일관성이 없는 글. 작성이 온전히 다 되지 않은 글. 등")
  ]
  random.shuffle(importance_items)
  importance_str = "\n".join([f"- {label} : {desc}" for label, desc in importance_items])

  # 2. department
  department_items = [
      ("교통실", "버스·지하철·택시, 대중교통 정책, 자전거·킥보드·보행, 주차, 신호, 불법주정차, 한강버스, 교통카드, 도로교통, 자율주행"),
      ("복지실", "기초생활보장, 저소득층 지원, 노숙인, 어르신 돌봄, 장애인 지원, 아동·청소년 복지, 한부모·다문화가족, 중장년 지원"),
      ("경제실", "창업·스타트업, 소상공인·전통시장 지원, 청년 취업·일자리, 중소기업 자금, 소비자 권익, 생활임금·노동정책, 자영업자, 지원금"),
      ("기후환경본부", "쓰레기·재활용, 소각장, 미세먼지·대기질, 동물보호, 탄소중립·신재생에너지, 친환경차·전기차 충전, 도시공원, 식품안전"),
      ("문화본부", "도서관, 박물관·문화시설, 공연·예술 지원, 문화유산, 전통문화, 관광 계획, 공원 시설 관리/조성"),
      ("시민건강국", "보건소, 응급의료, 감염병·방역, 정신건강, 예방접종, 치매 예방, 공중위생, 건강증진, 마약 대응, 금연 지원, 금연구역 관리"),
      ("재난안전실", "재난대응, 취약시설 점검, 도로·보도 안전, 대피소, 시민안전보험, 제설, 인파 안전관리, 도로공사 안전 관리, 공사현장 관리"),
      ("주택실", "재개발·재건축, 공공주택, 전세사기, 건축인허가, 도시계획, 주거환경개선, 도시재생, 시설물 관리, 공공시설 관리, 부동산, 사유지"),
      ("여성가족실", "보육·어린이집, 저출생 대응, 아동학대 예방, 청소년 지원·보호, 성폭력·성희롱 예방, 디지털성범죄, 여성 안전, 양성평등"),
      ("분류 보류", "정부 부서 관할 이외의 기관에 대한 내용. 정치적인 내용."),
  ]
  random.shuffle(department_items)
  department_str = "\n".join([f"- {label} : {desc}" for label, desc in department_items])

  # 3. complaint_type
  complaint_type_items = [
      ("신고", "불법 행위, 위험 상황, 규정 위반 등 제3자나 시설에 대한 문제를 알리는 경우"),
      ("문의", "제도, 정책, 절차, 방법 등에 대한 정보나 안내를 요청하는 경우"),
      ("건의", "정책 개선, 시설 설치, 제도 변경 등을 제안하는 경우"),
      ("항의", "행정 처리나 처우에 대한 불만을 표출하거나 시정을 요구하는 경우"),
      ("칭찬", "공무원, 서비스, 정책 등에 대한 긍정적인 평가를 담은 경우."),
      ("그 외", "위 유형 중 어느 것으로도 분류되지 않는 경우.")
  ]
  random.shuffle(complaint_type_items)
  complaint_type_str = "\n".join([f"- {label} : {desc}" for label, desc in complaint_type_items])

  # 4. emotion
  emotion_items = ["긍정", "중립", "부정"]
  random.shuffle(emotion_items)
  emotion_str = ", ".join(emotion_items)

  SYSTEM_PROMPT = f"""SYSTEM_PROMPT = 당신은 서울시 민원 분류 담당관입니다. 지금부터 민원과 민원에 대한 답변을 읽고 키워드를 추출해주세요.
민원은 제목인 Title과 본문인 Question으로 구분되어 입력됩니다.
이때 파이썬의 Dictionary 형태로 반환하세요.

1. importance
Title과 Question을 보고 해당 민원의 중요도를 파악해 {', '.join([item[0] for item in importance_items])} 중 레이블을 구분하세요.
{importance_str}

2. department
다음은 서울시의 각 부서가 담당하는 분야입니다. 민원 내용을 보고 해당 민원이 전달되어야 할 부서를 골라주세요.

{department_str}

단, Answer를 제외한 민원(Title과 Question)을 보았을 때 다음의 경우에 해당한다면 '분류 보류'를 설정하세요.
- Title과 Question만으로 민원의 주제를 알 수 없어 특정 부서를 분류할 수 없는 경우
- 첨부 파일을 업로드 했다고 되어 있으나 Title과 Question만으로 어떤 내용인지 유추할 수 없는 경우.
- Title과 Question이 내용을 알 수 없을 정도로 짧은 경우.

3. complaint_type
Title과 Question을 보고 민원의 유형을 다음 중 하나로 구분하세요
{complaint_type_str}

4. emotion
Title과 Question을 보고 민원인의 감정상태를 {emotion_str} 중 하나로 구분하세요.
"""
  return SYSTEM_PROMPT
# %%
def format_data(sample):

    return {
        "messages": [
            {
                "role": "system",
                "content": get_system_prompt(),
            },
            {
                "role": "user",
                "content": f"Title: {sample['title']}\nQuestion : {sample['Question']}",
            },
            {
                "role": "assistant",
                "content": str(sample['assistant'])
            },
        ],
    }

# train_df와 test_df를 OpenAI format으로 변환
train_dataset = []
for _, row in train_df.iterrows():
    train_dataset.append(format_data(row))

test_dataset = []
for _, row in test_df.iterrows():
    test_dataset.append(format_data(row))

# 최종 데이터셋 크기 출력
print(f"\n전체 데이터 분할 결과: Train {len(train_dataset)}개, Test {len(test_dataset)}개")
# %%
train_dataset[345]
# %%
# 리스트 형태에서 다시 Dataset 객체로 변경
print(type(train_dataset))
print(type(test_dataset))
train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
print(type(train_dataset))
print(type(test_dataset))
# %%
train_dataset[0]
# %% [markdown]
# ## 2. 모델 로드 및 템플릿 적용
# %%
# 허깅페이스 모델 ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# %%
tokenizer.padding_side = "left" #추론 시 padding left 로 설정되어있어야 함.
# %%
# 템플릿 적용
text = tokenizer.apply_chat_template(
    train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)
print(text)
# %% [markdown]
# ## 3. LoRA와 SFTConfig 설정
# %%
peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
)
# %% [markdown]
# `lora_alpha`: LoRA(Low-Rank Adaptation)에서 사용하는 스케일링 계수를 설정합니다. LoRA의 가중치 업데이트가 모델에 미치는 영향을 조정하는 역할을 하며, 일반적으로 학습 안정성과 관련이 있습니다.
# 
# `lora_dropout`: LoRA 적용 시 드롭아웃 확률을 설정합니다. 드롭아웃은 과적합(overfitting)을 방지하기 위해 일부 뉴런을 랜덤하게 비활성화하는 정규화 기법입니다. `0.1`로 설정하면 학습 중 10%의 뉴런이 비활성화됩니다.
# 
# `r`: LoRA의 랭크(rank)를 설정합니다. 이는 LoRA가 학습할 저차원 공간의 크기를 결정합니다. 작은 값일수록 계산 및 메모리 효율이 높아지지만 모델의 학습 능력이 제한될 수 있습니다.
# 
# `bias`: LoRA 적용 시 편향(bias) 처리 방식을 지정합니다. `"none"`으로 설정하면 편향이 LoRA에 의해 조정되지 않습니다. `"all"` 또는 `"lora_only"`와 같은 값으로 변경하여 편향을 조정할 수도 있습니다.
# 
# `target_modules`: LoRA를 적용할 특정 모듈(레이어)의 이름을 리스트로 지정합니다. 예제에서는 `"q_proj"`와 `"v_proj"`를 지정하여, 주로 Self-Attention 메커니즘의 쿼리와 값 프로젝션 부분에 LoRA를 적용합니다.
# 
# `task_type`: LoRA가 적용되는 작업 유형을 지정합니다. `"CAUSAL_LM"`은 Causal Language Modeling, 즉 시퀀스 생성 작업에 해당합니다. 다른 예로는 `"SEQ2SEQ_LM"`(시퀀스-투-시퀀스 언어 모델링) 등이 있습니다.
# %%
max_seq_length=8192

args = SFTConfig(
    output_dir="qwen2.5-0.5b",           # 저장될 디렉토리와 저장소 ID
    num_train_epochs=3,                      # 학습할 총 에포크 수
    per_device_train_batch_size=2,           # GPU당 배치 크기
    gradient_accumulation_steps=2,           # 그래디언트 누적 스텝 수
    gradient_checkpointing=True,             # 메모리 절약을 위한 체크포인팅
    optim="adamw_torch_fused",               # 최적화기
    logging_steps=10,                        # 로그 기록 주기
    save_strategy="steps",                   # 저장 전략
    save_steps=50,                           # 저장 주기
    bf16=True,                              # bfloat16 사용
    learning_rate=1e-4,                     # 학습률
    max_grad_norm=0.3,                      # 그래디언트 클리핑
    warmup_ratio=0.03,                      # 워밍업 비율
    lr_scheduler_type="constant",           # 고정 학습률
    push_to_hub=False,                      # 허브 업로드 안 함
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None,
    max_seq_length=max_seq_length,  # 최대 시퀀스 길이 설정
)
# %% [markdown]
# `output_dir`: 학습 결과가 저장될 디렉토리 또는 모델 저장소의 이름을 지정합니다. 이 디렉토리에 학습된 모델 가중치, 설정 파일, 로그 파일 등이 저장됩니다.
# 
# `num_train_epochs`: 모델을 학습시키는 총 에포크(epoch) 수를 지정합니다. 에포크는 학습 데이터 전체를 한 번 순회한 주기를 의미합니다. 예를 들어, `3`으로 설정하면 데이터셋을 3번 학습합니다.
# 
# `per_device_train_batch_size`: GPU 한 대당 사용되는 배치(batch)의 크기를 설정합니다. 배치 크기는 모델이 한 번에 처리하는 데이터 샘플의 수를 의미합니다. 작은 크기는 메모리 사용량이 적지만 학습 시간이 증가할 수 있습니다.
# 
# `gradient_accumulation_steps`: 그래디언트를 누적할 스텝(step) 수를 지정합니다. 이 값이 2로 설정된 경우, 두 스텝마다 그래디언트를 업데이트합니다. 배치 크기를 가상으로 늘리는 효과가 있으며, GPU 메모리 부족 문제를 해결할 때 유용합니다.
# 
# `gradient_checkpointing`: 그래디언트 체크포인팅을 활성화하여 메모리를 절약합니다. 이 옵션은 계산 그래프를 일부 저장하지 않고 다시 계산하여 메모리를 절약하지만, 속도가 약간 느려질 수 있습니다.
# 
# `optim`: 학습 시 사용할 최적화 알고리즘을 설정합니다. `adamw_torch_fused`는 PyTorch의 효율적인 AdamW 최적화기를 사용합니다.
# 
# `logging_steps`: 로그를 기록하는 주기를 스텝 단위로 지정합니다. 예를 들어, `10`으로 설정하면 매 10 스텝마다 로그를 기록합니다.
# 
# `save_strategy`: 모델을 저장하는 전략을 설정합니다. `"steps"`로 설정된 경우, 지정된 스텝마다 모델이 저장됩니다.
# 
# `save_steps`: 모델을 저장하는 주기를 스텝 단위로 설정합니다. 예를 들어, `50`으로 설정하면 매 50 스텝마다 모델을 저장합니다.
# 
# `bf16`: bfloat16 정밀도를 사용하도록 설정합니다. bfloat16은 FP32와 유사한 범위를 제공하면서 메모리와 계산 효율성을 높입니다.
# 
# `learning_rate`: 학습률을 지정합니다. 학습률은 모델의 가중치가 한 번의 업데이트에서 얼마나 크게 변할지를 결정합니다. 일반적으로 작은 값을 사용하여 안정적인 학습을 유도합니다.
# 
# `max_grad_norm`: 그래디언트 클리핑의 임계값을 설정합니다. 이 값보다 큰 그래디언트가 발생하면, 임계값으로 조정하여 폭발적 그래디언트를 방지합니다.
# 
# `warmup_ratio`: 학습 초기 단계에서 학습률을 선형으로 증가시키는 워밍업 비율을 지정합니다. 학습의 안정성을 높이기 위해 사용됩니다.
# 
# `lr_scheduler_type`: 학습률 스케줄러의 유형을 설정합니다. `"constant"`는 학습률을 일정하게 유지합니다.
# 
# `push_to_hub`: 학습된 모델을 허브에 업로드할지 여부를 설정합니다. `False`로 설정하면 업로드하지 않습니다.
# 
# `remove_unused_columns`: 사용되지 않는 열을 제거할지 여부를 설정합니다. True로 설정하면 메모리를 절약할 수 있습니다.
# 
# `dataset_kwargs`: 데이터셋 로딩 시 추가적인 설정을 전달합니다. 예제에서는 `skip_prepare_dataset: True`로 설정하여 데이터셋 준비 단계를 건너뜹니다.
# 
# `report_to`: 학습 로그를 보고할 대상을 지정합니다. `None`으로 설정되면 로그가 기록되지 않습니다.
# %% [markdown]
# ## 4. 학습 중 전처리 함수: collate_fn
# 배치 내의 데이터를 처리해 모델이 사용할 수 있는 입력 형식으로 변환하는 함수 <br>
# - input_ids, attention_mask : 메세지에서 개행 문자 제거하고 토큰화해 생성. input_ids 는 전체 토큰화 결과, attention_mask는 padding을 제외한 유효한 곳을 1로 표현한 데이터.
# - label : assistant 특수 토큰 (<|im_start|>assistant ) 이후로부터 <|im_end|>까지를 찾은 후 나머지는 -100으로 설정. -100은 loss 계산에서 제외.
# 
# -> 패딩 작업<br>
# - input_ids 패딩 token ID 추가
# - attention_mask에는 0 추가
# - label에는 -100 추가<br>
# 
# -> PyTorch tensor로 변환 반환.
# 
# %%
def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for example in batch:
        # messages의 각 내용에서 개행문자 제거
        clean_messages = []
        for message in example["messages"]:
            clean_message = {
                "role": message["role"],
                "content": message["content"]
            }
            clean_messages.append(clean_message)

        # 깨끗해진 메시지로 템플릿 적용
        text = tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=False
        ).strip()

        # 텍스트를 토큰화
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 레이블 초기화
        labels = [-100] * len(input_ids)

        # assistant 응답 부분 찾기
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        assistant = "assistant"

        # 토큰 ID 가져오기
        im_start_tokens = tokenizer.encode(im_start, add_special_tokens=False)
        im_end_tokens = tokenizer.encode(im_end, add_special_tokens=False)
        assistant_tokens = tokenizer.encode(assistant, add_special_tokens=False)

        i = 0
        while i < len(input_ids):
            # <|im_start|>assistant 찾기
            if (i + len(im_start_tokens) <= len(input_ids) and
                input_ids[i:i+len(im_start_tokens)] == im_start_tokens):

                # assistant 토큰 찾기
                assistant_pos = i + len(im_start_tokens)
                if (assistant_pos + len(assistant_tokens) <= len(input_ids) and
                    input_ids[assistant_pos:assistant_pos+len(assistant_tokens)] == assistant_tokens):

                    # assistant 응답의 시작 위치로 이동
                    current_pos = assistant_pos + len(assistant_tokens)

                    # <|im_end|>를 찾을 때까지 레이블 설정
                    while current_pos < len(input_ids):
                        if (current_pos + len(im_end_tokens) <= len(input_ids) and
                            input_ids[current_pos:current_pos+len(im_end_tokens)] == im_end_tokens):
                            # <|im_end|> 토큰도 레이블에 포함
                            for j in range(len(im_end_tokens)):
                                labels[current_pos + j] = input_ids[current_pos + j]
                            break
                        labels[current_pos] = input_ids[current_pos]
                        current_pos += 1

                    i = current_pos

            i += 1

        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attention_mask)
        new_batch["labels"].append(labels)

    # 패딩 적용
    max_length = max(len(ids) for ids in new_batch["input_ids"])

    for i in range(len(new_batch["input_ids"])):
        padding_length = max_length - len(new_batch["input_ids"][i])

        new_batch["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
        new_batch["attention_mask"][i].extend([0] * padding_length)
        new_batch["labels"][i].extend([-100] * padding_length)

    # 텐서로 변환
    for k, v in new_batch.items():
        new_batch[k] = torch.tensor(v)

    return new_batch
# %%
# collate_fn 테스트 (배치 크기 1로)
example = train_dataset[0]
batch = collate_fn([example])

print("\n처리된 배치 데이터:")
print("입력 ID 형태:", batch["input_ids"].shape)
print("어텐션 마스크 형태:", batch["attention_mask"].shape)
print("레이블 형태:", batch["labels"].shape)
# %%
print('입력에 대한 정수 인코딩 결과:')
print(batch["input_ids"][0].tolist())

print('레이블에 대한 정수 인코딩 결과:')
print(batch["labels"][0].tolist())

print('레이블에 대한 attention_mask 결과:')
print(batch["attention_mask"][0].tolist())
# %% [markdown]
# ## 5. 학습
# %%
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)
# %%
# 학습 시작
trainer.train()   # 모델이 자동으로 허브와 output_dir에 저장됨

# 모델 저장
trainer.save_model()   # 최종 모델을 저장
# %% [markdown]
# train 과정 loss dataframe으로 저장, 시각화
# %%
log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv('trainer_state.csv', index=False)

log_df.head(3)
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

loss_df = log_df[["step", "loss"]].dropna()
ax.plot(loss_df["step"], loss_df["loss"], color="steelblue", marker="o", markersize=3)
ax.set_title("Training Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()
# %% [markdown]
# ## 6. 테스트 데이터 준비
# %% [markdown]
# 실제 모델에 입력을 넣을 때에는 입력의 뒤에 '<|im_start|>assistant'가 부착되어서 넣는 것이 좋습니다. 그래야만 모델이 바로 답변을 생성합니다.
# %%
prompt_lst = []
label_lst = []

for prompt in test_dataset["messages"]:
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=False
    )
    input = text.split('<|im_start|>assistant')[0] + '<|im_start|>assistant'
    label = text.split('<|im_start|>assistant')[1]
    prompt_lst.append(input)
    label_lst.append(label)
# %%
print(prompt_lst[42])
# %%
print(label_lst[42])
# %% [markdown]
# ## 7. 파인 튜닝 모델 출력해보기
# %% [markdown]
# `AutoPeftModelForCausalLM` : LoRA Adapter와 기존의 LLM을 부착해 로드.
# - Fine-Tuning 가중치 저장된 checkpoint 경로 입력.
# - device_map='auto'로 자동으로 GPU에 배치
# 
# `pipeline` : NLP 작업을 간단히 할 수 있게 해주는 HuggingFace 고수준 유틸.
# - text-generation 입력으로 텍스트 생성 작업을 수행하기 위한 파이프라인 객체 생성.
# - 모델, 토크나이저 관리 - 입력 텍스트 토큰화 - 모델 입력 후 출력 생성 - 디코딩 다 하는 파이프라인임.
# %%
import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline
# %%
peft_model_id = "qwen2.5-0.5b/checkpoint-1450"
fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
# %%
eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]
# %%
def test_inference(pipe, prompt):
    outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=eos_token, do_sample=False)
    return outputs[0]['generated_text'][len(prompt):].strip()
# %%
for prompt, label in zip(prompt_lst[20:30], label_lst[20:30]):
    # print(f"    prompt:\n{prompt}")
    # print(f"    input:\n{prompt}")
    # print("-"*50)
    print(f"    response:\n{test_inference(pipe, prompt)}")
    # print(f"    label:\n{label}")
    print("="*50)
# %% [markdown]
# ## 8. 기본 모델 출력해보기
# %% [markdown]
# 이번에는 LoRA Adapter를 merge하지 않은 기본 모델로 테스트 데이터에 대해서 인퍼런스해보겠습니다.
# %%
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# %%
for prompt, label in zip(prompt_lst[20:30], label_lst[20:30]):
    # print(f"    prompt:\n{prompt}")
    print(f"response:{test_inference(pipe, prompt)}")
    # print(f"label:{label}")
    print("-"*50)
# %% [markdown]
# # 9. Fine-tuned vs Base 모델 정확도 평가
# %%
import ast
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


KEYS = ["importance", "department", "complaint_type", "emotion"]

#assistant 결과(str)로부터 각 key값 파싱(dict).
def parse_output(text):
    text = text.replace("<|im_end|>", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    result = {}

    for key in KEYS:
        match = re.search(r"['\"]?" + key + r"['\"]?\s*:\s*['\"]([^'\"]+)['\"]" , text)
        if match:
            result[key] = match.group(1)
    return result if result else None

def evaluate_model(pipe, prompt_lst, label_lst, model_name="Model"):                                                                                                                                   
      correct = {k: 0 for k in KEYS}
      total   = {k: 0 for k in KEYS}
      parse_fail = 0

      # 배치로 한번에 추론
      outputs = pipe(prompt_lst, max_new_tokens=1024, eos_token_id=eos_token, do_sample=False, batch_size=16)

      for output, prompt, label in zip(outputs, prompt_lst, label_lst):
          label_parsed = parse_output(label)
          if not label_parsed or not isinstance(label_parsed, dict):
              parse_fail += 1
              continue

          generated = output[0]['generated_text'][len(prompt):].strip()
          output_parsed = parse_output(generated)
          if not output_parsed or not isinstance(output_parsed, dict):
              parse_fail += 1
              continue

          for key in KEYS:
              if key in label_parsed and key in output_parsed:
                  total[key] += 1
                  if label_parsed[key] == output_parsed[key]:
                      correct[key] += 1

      accuracy = {k: (correct[k] / total[k] * 100) if total[k] > 0 else 0 for k in KEYS}
      print(f"\n[{model_name}] 파싱 실패: {parse_fail}건")
      for k in KEYS:
          print(f"  {k}: {correct[k]}/{total[k]} = {accuracy[k]:.1f}%")
      return accuracy

# # 1. Fine-tuned 모델 평가
# peft_model_id = "qwen2.5-0.5b/checkpoint-950"
# fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
#     peft_model_id, device_map="auto", torch_dtype=torch.float16
# )
# pipe_ft = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
# acc_ft = evaluate_model(pipe_ft, prompt_lst, label_lst, model_name="Fine-tuned")

# # 메모리 해제 : Fine-Tuned 모델 올라가 있는 동시에 Base 모델까지 올라가 메모리 부족하게 되는 현상 방지
# del fine_tuned_model, pipe_ft
# torch.cuda.empty_cache()

# 2. Base 모델 평가
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype=torch.float16
)
pipe_base = pipeline("text-generation", model=base_model, tokenizer=tokenizer)
acc_base = evaluate_model(pipe_base, prompt_lst[:30], label_lst[:30], model_name="Base")
# %%
#이후 checkpoint 별로 전부 키워드 정확도 도출 후 full_results_checkpoints.json으로 저장