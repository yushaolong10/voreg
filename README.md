# VoReg - 语音服务 API

基于 SpeechBrain 的声纹识别与语音分离服务，以及基于 BiLSTM+Attention 的句末检测（EOU）服务。

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

## 启动服务

```bash
bash run.sh
```

服务将在 `http://0.0.0.0:8000` 启动。

## 项目结构

```
voreg/
├── server.py              # 主入口，注册所有路由
├── router/
│   ├── voice.py           # 语音处理接口（声纹提取、比对、分离）
│   └── eou.py             # EOU 句末检测接口
├── train/
│   ├── train.py           # EOU 模型训练脚本
│   └── negative_samples.py # 负样本生成脚本
├── datasets/
│   ├── train.csv          # 训练数据
│   ├── test.csv           # 测试数据
│   └── readme.md          # 数据集说明
├── model                  # 训练好的模型
└── test.py                # 接口测试脚本
```

## API 接口

### 语音处理接口

#### 1. 提取声纹特征

**POST** `/v1/voice/embed`

提取音频的声纹特征向量。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pcm_base64 | string | 是 | 16kHz 单声道 int16 PCM 音频数据，base64 编码 |
| sample_rate | int | 否 | 采样率，默认 16000 |

**请求示例：**

```json
{
  "pcm_base64": "SGVsbG8gV29ybGQ=",
  "sample_rate": 16000
}
```

**响应示例：**

```json
{
  "emb": [0.123, -0.456, 0.789, ...]
}
```

#### 2. 声纹比对

**POST** `/v1/voice/score`

将音频与已有的声纹特征进行比对，返回相似度分数。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| ref_emb | list[float] | 是 | 参考声纹特征向量（从 /v1/voice/embed 获取） |
| pcm_base64 | string | 是 | 16kHz 单声道 int16 PCM 音频数据，base64 编码 |
| sample_rate | int | 否 | 采样率，默认 16000 |

**请求示例：**

```json
{
  "ref_emb": [0.123, -0.456, 0.789, ...],
  "pcm_base64": "SGVsbG8gV29ybGQ=",
  "sample_rate": 16000
}
```

**响应示例：**

```json
{
  "score": 0.85
}
```

**分数说明：**
- 分数范围：-1.0 ~ 1.0（余弦相似度）
- 分数越高表示声纹越相似
- 建议阈值：0.25 ~ 0.35（可根据实际场景调整）

#### 3. 语音分离提取

**POST** `/v1/voice/extract`

从混合音频中分离并提取目标说话人的语音。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| pcm_base64 | string | 是 | 16kHz 单声道 int16 PCM 混合音频，base64 编码 |
| ref_emb | list[float] | 是 | 目标说话人的声纹特征向量 |
| sample_rate | int | 否 | 采样率，默认 16000 |
| return_debug | bool | 否 | 是否返回调试信息，默认 false |

**响应示例：**

```json
{
  "target_pcm_int16": [123, -456, 789, ...],
  "score": 0.78,
  "src0_score": 0.32,
  "src1_score": 0.78,
  "picked": 1
}
```

---

### 句末检测接口 (EOU)

#### 4. 句末检测

**POST** `/v1/text/eou`

检测文本是否为完整句子（End-of-Utterance），用于判断用户是否说完了。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 待检测的文本 |
| threshold | float | 否 | 判断阈值，默认 0.6，概率 >= 阈值则判断为"说完了" |

**请求示例：**

```json
{
  "text": "你好，今天天气怎么样？",
  "threshold": 0.5
}
```

**响应示例：**

```json
{
  "text": "你好，今天天气怎么样？",
  "probability": 0.8234,
  "is_end": true
}
```

**字段说明：**
- `probability`: 模型预测的"说完了"概率，范围 0~1
- `is_end`: 是否判断为说完了（probability >= threshold）

**使用场景：**
- 语音交互中判断用户是否说完，避免"早切"（用户还没说完就被打断）
- 阈值越高越保守（减少早切，但可能漏检）

---

## EOU 模型训练

### 训练命令

```bash
python train/train.py
```

训练完成后会在项目根目录生成 `best.pt` 模型文件。

### 模型架构

**BiLSTM + Attention Pooling**

```
输入文本 → 字符编码 → Embedding(128) → BiLSTM(3层, hidden=128) → Attention Pooling → FC → sigmoid
```

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_len | 64 | 最大文本长度（保留末尾64个字符）|
| batch_size | 64 | 批次大小 |
| lr | 1e-3 | 初始学习率 |
| epochs | 30 | 最大训练轮数 |
| layers | 3 | LSTM 层数 |
| dropout | 0.1 | Dropout 比例 |
| threshold | 0.5 | 推理阈值 |

### 训练优化

- **学习率调度**：CosineAnnealingLR，从 1e-3 平滑衰减到 1e-6
- **类别不平衡处理**：根据正负样本比例自动计算 pos_weight
- **模型选择**：基于训练集 loss 最小值保存最佳模型
- **正则化**：weight_decay=1e-4

### 评估指标

| 指标 | 说明 |
|------|------|
| Precision | 预测为"说完"的准确率 |
| Recall | 真正说完的句子被召回的比例 |
| FPR | 误报率（早切率），没说完被判为说完的比例 |
| F1 | Precision 和 Recall 的调和平均 |

### 注意事项

⚠️ **模型参数必须保持一致！**

训练时 (`train/train.py`) 和推理时 (`router/eou.py`) 的模型参数必须完全一致：

```python
BiLSTMEOU(vocab_size=len(vocab), layers=3, dropout=0.1)
```

如果参数不匹配，会导致权重加载错误，所有预测都会偏向一侧。

---

## 测试

```bash
# 启动服务后运行测试
python test.py
```

测试脚本包含：
- 语音接口测试（embed、score、extract）
- EOU 句末检测测试

---

## 技术栈

- **FastAPI** - Web 框架
- **SpeechBrain** - 语音处理库
- **ECAPA-TDNN** - 声纹识别模型（spkrec-ecapa-voxceleb）
- **SepFormer** - 语音分离模型（sepformer-wsj02mix）
- **BiLSTM + Attention** - 句末检测模型
- **PyTorch** - 深度学习框架
