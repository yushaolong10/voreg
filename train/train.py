import csv
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# 1) 数据读取与词表
# -----------------------
def read_csv(path: str) -> List[Tuple[str, int]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=",")
        for row in r:
            text = row["text"].strip()
            label = int(row["label"])
            rows.append((text, label))
    return rows


def build_vocab(samples: List[Tuple[str, int]]) -> Dict[str, int]:
    chars = set()
    for text, _ in samples:
        for ch in text:
            chars.add(ch)
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    # 保留末尾 max_len 个字符
    text = text[-max_len:]
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in text]
    # 左 padding 到 max_len
    if len(ids) < max_len:
        pad = [vocab["<pad>"]] * (max_len - len(ids))
        ids = pad + ids
    return ids


# -----------------------
# 2) Dataset
# -----------------------
class EOUDataset(Dataset):
    def __init__(self, samples, vocab, max_len=64):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        x = torch.tensor(encode_text(text, self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


# -----------------------
# 3) 模型：BiLSTM + Attention Pooling
# -----------------------
class AttnPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask):
        # h: [B, T, H], mask: [B, T] (1 for valid, 0 for pad)
        scores = self.w(h).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        v = torch.sum(h * alpha, dim=1)  # [B, H]
        return v


class BiLSTMEOU(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim=128, hidden=128, layers=3, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.pool = AttnPooling(hidden_dim=hidden * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B, T]
        mask = (x != 0).long()  # pad=0
        e = self.embed(x)  # [B, T, E]
        h, _ = self.lstm(e)  # [B, T, 2H]
        v = self.pool(h, mask)  # [B, 2H]
        logits = self.fc(v)  # [B, 1]
        return logits


# -----------------------
# 4) 训练与评估
# -----------------------
@dataclass
class TrainCfg:
    max_len: int = 64
    batch_size: int = 64
    lr: float = 1e-3  # 初始学习率稍高，配合 scheduler 衰减
    epochs: int = 50
    threshold: float = 0.5
    patience: int = 10  # 早停耐心值


def split_data(samples, seed=42):
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.9)
    n_val = int(n * 0.1)
    return (
        samples[:n_train],
        samples[n_train:],
    )


def eval_model(model, loader, threshold=0.65, device="cpu"):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = torch.sigmoid(model(x))
            pred = (p >= threshold).long()
            y_int = y.long()
            tp += int(((pred == 1) & (y_int == 1)).sum())
            fp += int(((pred == 1) & (y_int == 0)).sum())
            tn += int(((pred == 0) & (y_int == 0)).sum())
            fn += int(((pred == 0) & (y_int == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)  # 对你来说：False End 近似看这个
    f1 = 2 * precision * recall / (precision + recall + 1e-9)  # F1 综合指标
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def train(train_path="train.csv", test_path="test.csv"):
    samples = read_csv(train_path)
    train_s, val_s = split_data(samples, 33)
    print("训练数量: 训练集：", len(train_s), "验证集：", len(val_s))

    vocab = build_vocab(train_s)
    cfg = TrainCfg()

    train_ds = EOUDataset(train_s, vocab, cfg.max_len)
    val_ds = EOUDataset(val_s, vocab, cfg.max_len)

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = BiLSTMEOU(vocab_size=len(vocab), layers=3).to(device)

    # 类不平衡时可以加 pos_weight
    n_pos = sum(1 for _, label in train_s if label == 1)
    n_neg = len(train_s) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"正样本: {n_pos}, 负样本: {n_neg}, pos_weight: {pos_weight.item():.2f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # 学习率调度器：余弦退火，让学习率平滑下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.epochs, eta_min=1e-6
    )

    best = None
    best_loss = 1

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item()

        # 更新学习率
        scheduler.step()
        current_lr = optim.param_groups[0]["lr"]

        m = eval_model(model, val_ld, threshold=cfg.threshold, device=device)
        avg_loss = total_loss / len(train_ld)
        print(
            f"epoch={epoch:2d} loss={avg_loss:.4f} lr={current_lr:.2e} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f}"
        )

        # 使用 F1 综合指标选择最佳模型
        if avg_loss < best_loss:
            best = m
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "vocab": vocab}, "./model/best.pt")

    print(f"\n训练完成! 最佳验证集结果: F1={best['f1']:.4f}")

    samples = read_csv(test_path)
    test_ds = EOUDataset(samples, vocab, cfg.max_len)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size)
    ckpt = torch.load("./model/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    m = eval_model(model, test_ld, threshold=cfg.threshold, device=device)
    print("test:", m)


if __name__ == "__main__":
    train(train_path="./datasets/train.csv", test_path="./datasets/test.csv")
    # train(train_path="./datasets/train.csv", test_path="./datasets/test_not1.csv")
    # train(train_path="./datasets/train.csv", test_path="./datasets/test_not2.csv")
    # train(train_path="./datasets/train.csv", test_path="./datasets/test_ok.csv")
