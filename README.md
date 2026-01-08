# VoReg - 声纹识别服务

基于 SpeechBrain ECAPA-TDNN 模型的声纹特征提取与比对服务。

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

## API 接口

### 1. 提取声纹特征

**POST** `/v1/embed`

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

### 2. 声纹比对

**POST** `/v1/score`

将音频与已有的声纹特征进行比对，返回相似度分数。

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| ref_emb | list[float] | 是 | 参考声纹特征向量（从 /v1/embed 获取） |
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

## 技术栈

- **FastAPI** - Web 框架
- **SpeechBrain** - 语音处理库
- **ECAPA-TDNN** - 声纹识别模型（spkrec-ecapa-voxceleb）
- **PyTorch** - 深度学习框架

