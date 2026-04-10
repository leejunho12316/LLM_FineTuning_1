

# 서울시 응답소 민원 Q&A 데이터셋

## 1. 데이터 출처

**서울시 응답소 - 시장에게 알린다**
https://eungdapso.seoul.go.kr/req/mayor_hope/mayor_hope.do

서울시 응답소 공식 홈페이지에 공개된 민원 데이터를 Web Crawling하여 수집.
시민이 서울시장에게 직접 민원·건의 사항을 올리면, 서울시가 답변하는 Q&A 형태의 공개 데이터.

<img src="./readme_res/1_eungdapso_screenshot.png">

## 2. 데이터셋 구조

| 컬럼명            | 설명              |
|----------------|-----------------|
| `title`        | 민원 제목           |
| `Date`         | 민원 접수 날짜        |
| `Question`     | 시민이 작성한 민원 내용   |
| `Answer`       | 서울시 답변 내용       |
| `rceptNo_enc`  | 암호화된 민원 고유 접수번호 |


## 3. 데이터셋 통계

| 구분 | 파일명 | 행 수 |
|---|---|---|
| 원본 (Raw) | `mayor_hope.csv` | 13,540 |
| 정제 후 (Cleaned) | `mayor_hope_cleaned.csv` | 13,184 |

원본 대비 **356건**의 이상 데이터(결측값, 중복, 특정 유형 등)를 제거.

- 결측값(NaN) 제거: `Question` 또는 `Answer`가 비어있는 행 삭제.
- 중복 데이터 제거: `Question` 중복 행 삭제 -> 도배글, 어그로성 글, 중복 비난글 다수.
- 기타 데이터 제거 : 첨부된 이미지나 파일을 강조하는 글이지만 서버/홈페이지의 문제로 유실되고 내용을 알 수 없는 데이터.
- 정제된 데이터는 `mayor_hope_cleaned.csv`로 저장


## 4. HuggingFace Dataset

<img src="https://camo.githubusercontent.com/e70f2a6a8c8f5bf0f4211dd32a0b5311c7464b65098006e654986f6738bfe034/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7261772f6d61696e2f68756767696e67666163655f6875622e737667">

Data Link : https://huggingface.co/datasets/leejunho12316/seoul-mayor-hope

huggingface 데이터셋 README 도 작성