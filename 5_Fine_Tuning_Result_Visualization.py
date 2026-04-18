################개요################
# 모델 checkpoint 별 키워드 종합 점수 시각화 코드

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── 1. 데이터 로드 ──────────────────────────────────────────────
JSON_PATH = "3.Fine_Tuning/full_results_checkpoints.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# checkpoint step → int 변환 후 정렬
records = []
for step, metrics in raw.items():
    row = {"step": int(step)}
    row.update(metrics)
    records.append(row)

df = pd.DataFrame(records).sort_values("step").reset_index(drop=True)

# long-form 으로 변환 (seaborn lineplot 용)
keyword_cols = ["importance", "department", "complaint_type", "emotion"]
df_long = df.melt(id_vars="step", value_vars=keyword_cols,
                  var_name="keyword", value_name="accuracy")

# ── 2. 스타일 설정 ──────────────────────────────────────────────
sns.set_theme(style="darkgrid", context="notebook", font_scale=1.15)

PALETTE = {
    "importance":     "#4C72B0",
    "department":     "#DD8452",
    "complaint_type": "#55A868",
    "emotion":        "#C44E52",
}
LABEL_MAP = {
    "importance":     "Importance",
    "department":     "Department",
    "complaint_type": "Complaint Type",
    "emotion":        "Emotion",
}

# ── 3. Figure 생성 ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

fig.suptitle("Fine-Tuning Accuracy by Checkpoint Step",
             fontsize=18, fontweight="bold", y=1.01)

for ax, kw in zip(axes, keyword_cols):
    sub = df_long[df_long["keyword"] == kw]
    color = PALETTE[kw]

    # 라인 + 마커
    sns.lineplot(
        data=sub, x="step", y="accuracy",
        color=color, linewidth=2.2, marker="o",
        markersize=5, ax=ax, label=LABEL_MAP[kw],
    )

    # 최고점 강조
    best_idx = sub["accuracy"].idxmax()
    best_step = sub.loc[best_idx, "step"]
    best_acc  = sub.loc[best_idx, "accuracy"]

    ax.scatter(best_step, best_acc, color=color,
               s=120, zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(
        f"Best\n{best_acc:.1f}% @ step {best_step}",
        xy=(best_step, best_acc),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
    )

    # 축 꾸미기
    ax.set_title(LABEL_MAP[kw], fontsize=13, fontweight="semibold", pad=8)
    ax.set_xlabel("Checkpoint Step", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlim(df["step"].min() - 20, df["step"].max() + 20)
    ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("./3.Fine_Tuning/4_Fine_Tuning_Accuracy_by_Checkpoint.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("그래프 저장 완료: 4_Fine_Tuning_Accuracy_by_Checkpoint.png")

# ── 4. 통합 그래프 (모든 키워드 한 화면) ───────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))

for kw in keyword_cols:
    sub = df_long[df_long["keyword"] == kw]
    sns.lineplot(
        data=sub, x="step", y="accuracy",
        color=PALETTE[kw], linewidth=2, marker="o",
        markersize=4, ax=ax2, label=LABEL_MAP[kw],
    )

ax2.set_title("Fine-Tuning Accuracy — All Keywords", fontsize=15, fontweight="bold")
ax2.set_xlabel("Checkpoint Step", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(150))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
ax2.legend(title="Keyword", fontsize=10, title_fontsize=10)

plt.tight_layout()
plt.savefig("./3.Fine_Tuning/4_Fine_Tuning_Accuracy_Combined.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("그래프 저장 완료: 4_Fine_Tuning_Accuracy_Combined.png")