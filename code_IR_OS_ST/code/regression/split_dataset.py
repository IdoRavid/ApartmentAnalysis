import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from common import parse_target, out_dir_for_target

TARGET = parse_target("reactionsCount")
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEV_SIZE_FROM_REMAINDER = 0.25  # 0.25 of 0.80 -> 0.20

src = Path("../../output/regression/dataset.csv")
out_dir = out_dir_for_target(TARGET)

df = pd.read_csv(src)


df = df[df[TARGET].notna()].copy()
y = df[TARGET].astype(float).fillna(0.0)


train_dev, test = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


train, dev = train_test_split(
    train_dev, test_size=DEV_SIZE_FROM_REMAINDER, random_state=RANDOM_STATE
)

train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8-sig")
dev.to_csv(out_dir / "dev.csv", index=False, encoding="utf-8-sig")
test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8-sig")
print(f"[{TARGET}] Saved: {len(train)} train, {len(dev)} dev, {len(test)} test to {out_dir}")
