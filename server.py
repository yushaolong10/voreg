# server.py
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation
from modelscope import snapshot_download


app = FastAPI()

snapshot_download('speechbrain/spkrec-ecapa-voxceleb')
snapshot_download('speechbrain/sepformer-wsj02mix')

# 1) ECAPA: speaker embedding
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},  # 服务端CPU即可
)

# 2) SepFormer: blind separation (2 speakers)
sep = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    run_opts={"device": "cpu"},
)


class EmbedReq(BaseModel):
    pcm_base64: str  # 16k mono int16 PCM, base64 encoded
    sample_rate: int = 16000


class ScoreReq(BaseModel):
    ref_emb: list[float]
    pcm_base64: str  # 16k mono int16 PCM, base64 encoded
    sample_rate: int = 16000


class TSSReq(BaseModel):
    pcm_base64: str  # 16k mono int16 PCM, base64 encoded
    ref_emb: list[float]
    sample_rate: int = 16000
    return_debug: bool = False


def pcm_base64_to_tensor(pcm_base64: str, check_length: bool = True):
    pcm_bytes = base64.b64decode(pcm_base64)
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(x).unsqueeze(0)  # [1, T]
    return wav


@app.post("/v1/embed")
def embed(req: EmbedReq):
    wav = pcm_base64_to_tensor(req.pcm_base64)
    with torch.no_grad():
        emb = classifier.encode_batch(wav).squeeze(0).squeeze(0)  # [D]
        emb = torch.nn.functional.normalize(emb, dim=0)
    return {"emb": emb.cpu().tolist()}


@app.post("/v1/score")
def score(req: ScoreReq):
    wav = pcm_base64_to_tensor(req.pcm_base64)
    ref = torch.tensor(req.ref_emb, dtype=torch.float32)
    ref = torch.nn.functional.normalize(ref, dim=0)
    with torch.no_grad():
        emb = classifier.encode_batch(wav).squeeze(0).squeeze(0)
        emb = torch.nn.functional.normalize(emb, dim=0)
    s = torch.dot(ref, emb).item()
    return {"score": float(s)}


@app.post("/v1/extract")
def tss_extract(req: TSSReq):
    mix = pcm_base64_to_tensor(req.pcm_base64)  # [1, T]

    ref = torch.tensor(req.ref_emb, dtype=torch.float32)
    ref = F.normalize(ref, dim=0)

    with torch.no_grad():
        # SepFormer separation: [B, T, N] where N=2
        est = sep.separate_batch(
            mix
        )  # SpeechBrain separation API :contentReference[oaicite:2]{index=2}
        src0 = est[0, :, 0].unsqueeze(0)  # [1, T]
        src1 = est[0, :, 1].unsqueeze(0)

        # ECAPA embedding for each separated source
        e0 = classifier.encode_batch(src0).squeeze()
        e1 = classifier.encode_batch(src1).squeeze()
        e0 = F.normalize(e0, dim=0)
        e1 = F.normalize(e1, dim=0)

        s0 = torch.dot(ref, e0).item()
        s1 = torch.dot(ref, e1).item()

        if s1 > s0:
            target = src1
            score = s1
            picked = 1
        else:
            target = src0
            score = s0
            picked = 0

    # float32 [-1,1] -> int16
    y = (target.squeeze(0).cpu().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)

    resp = {
        "target_pcm_int16": y.tolist(),
        "score": float(score),
    }
    if req.return_debug:
        resp.update(
            {"src0_score": float(s0), "src1_score": float(s1), "picked": picked}
        )
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
