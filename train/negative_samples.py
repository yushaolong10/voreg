import argparse
import csv
import itertools
import random
from typing import List, Set


# 生成负样本脚本，用于补充负样本数据
# -----------------------------
# 词库（可按你业务扩充）
# -----------------------------
SUBJ = ["我", "我们", "咱们", "这边", "你们", "他们"]
INTENT = ["想", "想要", "准备", "打算", "需要", "希望", "计划", "考虑", "想问", "想确认", "想了解"]
VERB = ["做", "弄", "处理", "看看", "查", "问", "改", "安排", "订", "买", "选", "找", "联系", "提交", "确认"]
OBJ = [
    "这个", "那个", "这些", "那些", "一件事", "一个问题", "这个问题", "那个问题", "方案", "计划", "安排", "事情",
    "订单", "地址", "时间", "价格", "信息", "结果", "进度", "细节", "原因", "情况", "需求", "选项", "步骤"
]
TIME = ["今天", "明天", "后天", "这周", "下周", "最近", "等会", "一会儿", "之后", "稍后"]
PLACE = ["在家", "在公司", "在上海", "在北京", "在那边", "在这边", "到家", "到公司", "去上海", "去北京"]
CONNECT = ["然后", "还有", "而且", "所以", "但是", "因为", "如果", "不过", "另外", "再说", "同时", "顺便"]
FUNC_TAIL = ["在", "对", "给", "把", "和", "跟", "用", "从", "到", "向", "为", "关于", "以及"]
HEDGE = ["那个", "就是", "大概", "可能", "先", "再", "稍微", "有点", "比较", "主要是"]
ENUM = ["比如", "像", "包括", "尤其是", "譬如说"]


# -----------------------------
# 模板（全部 label=0：未结束）
# 设计目标：看起来像句子，但明显还要继续
# -----------------------------
def build_candidates() -> List[str]:
    cands: Set[str] = set()

    # 1) 意图前缀：我想 / 我想要 + (动词/对象/时间/地点) 的不完整组合
    for s, it in itertools.product(SUBJ, INTENT):
        cands.add(f"{s}{it}")
        for o in OBJ:
            cands.add(f"{s}{it}{o}")
        for v in VERB:
            cands.add(f"{s}{it}{v}")
        for v, o in itertools.product(VERB, OBJ):
            cands.add(f"{s}{it}{v}{o}")
        for t in TIME:
            cands.add(f"{s}{it}{t}")
        for p in PLACE:
            cands.add(f"{s}{it}{p}")

    # 2) 连词/转折开头：然后/但是/因为/如果 + (主语/意图/动词...)（典型未完）
    for c in CONNECT:
        cands.add(f"{c}")
        for s in SUBJ:
            cands.add(f"{c}{s}")
        for s, it in itertools.product(SUBJ, INTENT):
            cands.add(f"{c}{s}{it}")
        for s, it, v in itertools.product(SUBJ, INTENT, VERB):
            cands.add(f"{c}{s}{it}{v}")

    # 3) 指代未消解：那个/这个/这些 + 名词
    for h, o in itertools.product(["那个", "这个", "这种", "那种", "这些", "那些"], OBJ):
        cands.add(f"{h}{o}")

    # 4) 功能词结尾：以介词/连词结尾几乎必未完
    for s, v, tail in itertools.product(SUBJ, VERB, FUNC_TAIL):
        cands.add(f"{s}{v}{tail}")
    for s, it, tail in itertools.product(SUBJ, INTENT, FUNC_TAIL):
        cands.add(f"{s}{it}{tail}")
    for s, it, v, tail in itertools.product(SUBJ, INTENT, VERB, FUNC_TAIL):
        cands.add(f"{s}{it}{v}{tail}")

    # 5) 列举/举例开头：比如/包括/像 + 对象（通常后面还会继续列）
    for e, o in itertools.product(ENUM, OBJ):
        cands.add(f"{e}{o}")
    for s, e in itertools.product(SUBJ, ENUM):
        cands.add(f"{s}{e}")

    # 6) 带语气填充词的半句：那个/就是 + 前缀
    for h, s, it in itertools.product(HEDGE, SUBJ, INTENT):
        cands.add(f"{h}{s}{it}")
        for o in OBJ:
            cands.add(f"{h}{s}{it}{o}")

    # 7) 组合 2 段（用顿号/逗号连接，但不加句号，制造“还要继续”的感觉）
    #   段1从候选里取一部分，段2从 CONNECT/INTENT 取
    base1 = list(cands)[:5000]  # 控制规模，避免爆炸
    for a, b in itertools.product(base1, CONNECT):
        cands.add(f"{a}，{b}")
    for a, s, it in itertools.product(base1[:2000], SUBJ, INTENT):
        cands.add(f"{a}，{s}{it}")

    # 清理：不要生成看起来完结的（句号/问号/叹号结尾）
    cands = {x.strip() for x in cands if x.strip() and not x.strip().endswith(("。", "！", "？", "?", "!", "."))}

    return sorted(cands)


def write_tsv(samples: List[str], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["text", "label"])
        for s in samples:
            w.writerow([s, 0])


def main():
    random.seed(42)
    NEG_NUM = 5000   # 👈 你可以调
    out_path = "negative_samples.csv"

    cands = build_candidates()
    total = len(cands)

    if NEG_NUM > total:
        raise SystemExit(f"Not enough unique candidates: requested {NEG_NUM}, but only {total} available. "
                         f"Expand vocab/templates to increase space.")

    # 随机抽样 n 条（不重复）
    picked = random.sample(cands, NEG_NUM)
    write_tsv(picked, out_path)

    print(f"Total unique candidates: {total}")
    print(f"Wrote {len(picked)} samples -> {out_path}")


if __name__ == "__main__":
    main()
