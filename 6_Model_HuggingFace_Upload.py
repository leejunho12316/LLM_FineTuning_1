import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# ── 설정 ──────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN   = os.getenv("HF_TOKEN")
HF_USER    = "leejunho12316"
REPO_NAME  = "qwen2.5-0.5b-finetuned-checkpoint1150"          # 원하는 repo 이름으로 변경
REPO_ID    = f"{HF_USER}/{REPO_NAME}"
PRIVATE    = True                               # True 로 바꾸면 비공개

CHECKPOINT_DIR = Path("3.Fine_Tuning/qwen2.5-0.5b/checkpoint-1150")

# 학습 재개용 파일은 제외 (추론에 불필요)
EXCLUDE_FILES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api = HfApi(token=HF_TOKEN)

    # 1. repo 생성 (이미 있으면 그냥 넘어감)
    create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        private=PRIVATE,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    # 2. 업로드할 파일 목록
    files = [f for f in CHECKPOINT_DIR.iterdir() if f.name not in EXCLUDE_FILES]
    print(f"\n업로드할 파일 ({len(files)}개):")
    for f in sorted(files):
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # 3. 파일 업로드
    print("\n업로드 시작...")
    for file_path in sorted(files):
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )
        print(f"  ✓ {file_path.name}")

    print(f"\n완료! 모델 페이지: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()