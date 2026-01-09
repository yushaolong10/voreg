# router/eou.py
"""
End-of-Utterance (EOU) 检测接口
使用训练好的 BiLSTM + Attention 模型判断句子是否说完
"""
import os
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn


router = APIRouter(prefix="/v1/text", tags=["eou"])

# 模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/best.pt")


# -----------------------
# 模型定义（与训练时保持一致）
# -----------------------
class AttnPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask):
        scores = self.w(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        v = torch.sum(h * alpha, dim=1)
        return v


class BiLSTMEOU(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim=128, hidden=128, layers=3, dropout=0.1
    ):
        """注意：layers 和 dropout 必须与训练时保持一致！"""
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
        mask = (x != 0).long()
        e = self.embed(x)
        h, _ = self.lstm(e)
        v = self.pool(h, mask)
        logits = self.fc(v)
        return logits


# -----------------------
# 文本编码（与训练时保持一致）
# -----------------------
def encode_text(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    """保留末尾 max_len 个字符，左 padding"""
    text = text[-max_len:]
    ids = [vocab.get(ch, vocab.get("<unk>", 1)) for ch in text]
    if len(ids) < max_len:
        pad = [vocab.get("<pad>", 0)] * (max_len - len(ids))
        ids = pad + ids
    return ids


# -----------------------
# 加载模型
# -----------------------
_model = None
_vocab = None


def get_model_and_vocab():
    global _model, _vocab
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"EOU 模型文件不存在: {MODEL_PATH}，请先运行 train/eou.py 训练模型",
            )
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _vocab = ckpt["vocab"]
        # 必须与训练时的参数完全一致：layers=3, dropout=0.1
        _model = BiLSTMEOU(vocab_size=len(_vocab), layers=3, dropout=0.1)
        _model.load_state_dict(ckpt["model"])
        _model.eval()
    return _model, _vocab


# -----------------------
# API 请求/响应模型
# -----------------------
class EOURequest(BaseModel):
    text: str  # 待检测的文本
    threshold: float = 0.6  # 判断阈值，默认 0.4


class EOUResponse(BaseModel):
    text: str  # 输入文本
    probability: float  # 模型预测的"说完了"概率
    is_end: bool  # 是否判断为说完了


# -----------------------
# API 接口
# -----------------------
@router.post("/eou", response_model=EOUResponse)
def detect_eou(req: EOURequest):
    """
    检测单条文本是否为句末（End-of-Utterance）

    - text: 待检测的文本
    - threshold: 判断阈值（默认 0.4），概率 >= 阈值则判断为"说完了"
    """
    model, vocab = get_model_and_vocab()

    # 编码文本
    ids = encode_text(req.text, vocab)
    x = torch.tensor([ids], dtype=torch.long)

    # 推理
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    is_end = prob >= req.threshold

    return EOUResponse(
        text=req.text,
        probability=round(prob, 4),
        is_end=is_end,
    )
