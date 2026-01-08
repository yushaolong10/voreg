import base64
import requests
import numpy as np

BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 16000
MIN_DURATION = 3  # 最小音频时长（秒），ECAPA-TDNN 需要足够长的音频


def generate_test_audio(duration: float = MIN_DURATION, freq: float = 440.0):
    """生成测试用正弦波音频"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * freq * t) * 0.5
    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    return audio_int16


def test_embed():
    """测试 /v1/embed 接口"""
    print("=" * 50)
    print("测试 /v1/embed")
    audio_data = generate_test_audio(duration=MIN_DURATION, freq=300)
    pcm_base64 = base64.b64encode(audio_data.tobytes()).decode()

    resp = requests.post(
        f"{BASE_URL}/v1/embed",
        json={"pcm_base64": pcm_base64, "sample_rate": SAMPLE_RATE},
    )
    result = resp.json()
    if "emb" not in result:
        print(f"错误: {result}")
        return None
    emb = result["emb"]
    print(f"声纹特征维度: {len(emb)}")
    print(f"声纹特征前5个值: {emb[:5]}")
    return emb


def test_score(ref_emb):
    """测试 /v1/score 接口"""
    print("=" * 50)
    print("测试 /v1/score")
    audio_data = generate_test_audio(duration=MIN_DURATION, freq=300)
    pcm_base64 = base64.b64encode(audio_data.tobytes()).decode()

    resp = requests.post(
        f"{BASE_URL}/v1/score",
        json={"ref_emb": ref_emb, "pcm_base64": pcm_base64, "sample_rate": SAMPLE_RATE},
    )
    result = resp.json()
    if "score" not in result:
        print(f"错误: {result}")
        return None
    score = result["score"]
    print(f"相似度分数: {score}")
    return score


def test_extract(ref_emb):
    """测试 /v1/extract 接口"""
    print("=" * 50)
    print("测试 /v1/extract")

    # 生成模拟混合音频（两个不同频率的正弦波）
    duration = max(MIN_DURATION, 3)  # 至少 3 秒
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

    # 模拟两个说话人的音频（用不同频率正弦波代替）
    speaker1 = np.sin(2 * np.pi * 300 * t) * 0.5  # 300Hz
    speaker2 = np.sin(2 * np.pi * 600 * t) * 0.5  # 600Hz
    mix = speaker1 + speaker2

    # 转换为 int16
    mix_int16 = (mix * 32768).clip(-32768, 32767).astype(np.int16)
    pcm_base64 = base64.b64encode(mix_int16.tobytes()).decode()

    resp = requests.post(
        f"{BASE_URL}/v1/extract",
        json={
            "ref_emb": ref_emb,
            "pcm_base64": pcm_base64,
            "sample_rate": SAMPLE_RATE,
            "return_debug": True,
        },
    )

    result = resp.json()
    if "score" not in result:
        print(f"错误: {result}")
        return None
    print("分离结果:")
    print(f"  - 匹配分数: {result['score']:.4f}")
    print(f"  - src0 分数: {result.get('src0_score', 'N/A')}")
    print(f"  - src1 分数: {result.get('src1_score', 'N/A')}")
    print(f"  - 选择的源: {result.get('picked', 'N/A')}")
    print(f"  - 输出音频长度: {len(result['target_pcm_int16'])} samples")
    return result


if __name__ == "__main__":
    # 1. 测试 embed
    emb = test_embed()

    # 2. 测试 score
    test_score(emb)

    # 3. 测试 extract
    test_extract(emb)
