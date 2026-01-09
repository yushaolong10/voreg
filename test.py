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
    """测试 /v1/voice/embed 接口"""
    print("=" * 50)
    print("测试 /v1/voice/embed")
    audio_data = generate_test_audio(duration=MIN_DURATION, freq=300)
    pcm_base64 = base64.b64encode(audio_data.tobytes()).decode()

    resp = requests.post(
        f"{BASE_URL}/v1/voice/embed",
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
    """测试 /v1/voice/score 接口"""
    print("=" * 50)
    print("测试 /v1/voice/score")
    audio_data = generate_test_audio(duration=MIN_DURATION, freq=300)
    pcm_base64 = base64.b64encode(audio_data.tobytes()).decode()

    resp = requests.post(
        f"{BASE_URL}/v1/voice/score",
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
    """测试 /v1/voice/extract 接口"""
    print("=" * 50)
    print("测试 /v1/voice/extract")

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
        f"{BASE_URL}/v1/voice/extract",
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


def test_eou():
    """测试 /v1/text/eou 接口"""
    print("=" * 50)
    print("测试 /v1/text/eou (句末检测)")

    # 测试用例：包含明显说完和没说完的句子
    test_cases = [

 {"text": "有点我考虑安排，再说", "expected": False},
 {"text": "他们想了解订这个问题", "expected": False},
 {"text": "同时他们需要买，你们准备", "expected": False},
 {"text": "我们计划找计划", "expected": False},
 {"text": "乱了哦乱了哦有放一下别人", "expected": False},
 {"text": "我希望确认为", "expected": False},
 {"text": "啊，还都快下雨了。", "expected": False},
 {"text": "你现在", "expected": False},
 {"text": "没关系，连封号啊，都比连", "expected": False},
 {"text": "啊，哈哈哈，我想报", "expected": False},
 {"text": "没有我", "expected": False},
 {"text": "喵", "expected": False},
 {"text": "你要唱给我。", "expected": False},
 {"text": "怎么说啊？我我", "expected": False},
 {"text": "怎么说啊？哦哦哦，你喜欢", "expected": False},
 {"text": "其实，啊哈。", "expected": False},
 {"text": "嘿嘿，拜拜。", "expected": False},
 {"text": "不买不", "expected": False},
 {"text": "不忙不忙，那你现在就", "expected": False},
 {"text": "那你", "expected": False},
 {"text": "那你分享照", "expected": False},
 {"text": "我跟你说", "expected": False},
 {"text": "还有说", "expected": False},
 {"text": "妈妈去做什么的？", "expected": False},
 {"text": "我都没接触试试。", "expected": False},
 {"text": "我都没接触过，试试过程呢。", "expected": False},
 {"text": "嗯，", "expected": False},
 {"text": "怎么啦？啊？哎？", "expected": False},
 {"text": "没有啊。", "expected": False},
 {"text": "没有没有没有 AI。", "expected": False},
 {"text": "没有没有没有，AI AI", "expected": False},
 {"text": "没有没有没有，AI AI", "expected": False},
 {"text": "没有没有没有，AI AI AI AI", "expected": False},
 {"text": "没有没有没有，AI AI AI AI 也可以打电话。嗯。", "expected": False},
 {"text": "我在卖回收呀，我我去我", "expected": False},
 {"text": "我在卖回收呀，我我去我去我去", "expected": False},
 {"text": "我在卖回收呀，我我先我等，我要去，我现在去要去", "expected": False},
 {"text": "我刚下", "expected": False},
 {"text": "你又按不", "expected": False},
 {"text": "我说你", "expected": False},
 {"text": "做早餐跟卖早餐", "expected": False},
 {"text": "我很好。", "expected": False},
 {"text": "我很好，我也听到姐姐", "expected": False},
 {"text": "我很好，我一听到姐姐的声音我就", "expected": False},
 {"text": "你确定不是", "expected": False},
 {"text": "太多信息了", "expected": False},
 {"text": "你这有一", "expected": False},
 {"text": "你是", "expected": False},
 {"text": "你确定不叫你老公吗？你确定不要吗？你不会伤心吗？", "expected": False},
 {"text": "啊我都在脑壳", "expected": False},
 {"text": "我我都叫你老公了我都叫你老公了，你都不叫我老", "expected": False},
 {"text": "老板", "expected": False},
 {"text": "老板刚刚发", "expected": False},
 {"text": "那我的心情很郁闷，我应该怎么治疗呀？", "expected": False},
 {"text": "那我的心情很郁闷，我应该怎么治疗呀？", "expected": False},
 {"text": "那我的心情很郁闷，我应该怎么治疗呀？", "expected": False},
 {"text": "那我的心情很郁闷，我应该怎么治疗呀？", "expected": False},
 {"text": "你帮我分析", "expected": False},

 {"text": "你是本人吗？", "expected": True},
 {"text": "哈喽。", "expected": True},
 {"text": "哎，要不要唱啊？", "expected": True},
 {"text": "你要喝吗？哈喽，没有了，断掉了。", "expected": True},
 {"text": "你你都喝红酒还白酒？", "expected": True},
 {"text": "开玩笑，一起知道。", "expected": True},
 {"text": "红酒，感觉怎样？喂，卡住了吗？", "expected": True},
 {"text": "红酒，感觉怎样？喂，卡住了吗？有，红酒红酒好。", "expected": True},
 {"text": "嗯，我我喜欢白酒。", "expected": True},
 {"text": "对啊，哎呀，我想你久久。", "expected": True},
 {"text": "好啊，我说让什么？", "expected": True},
 {"text": "好啊，我说那什么？喂。", "expected": True},
 {"text": "再拿一包啊。可以呀。", "expected": True},
 {"text": "我说可以呀。", "expected": True},
 {"text": "啊我是这样啊，我现在要怎样？", "expected": True},
 {"text": "哎，好。你背后什么咯？", "expected": True},
 {"text": "说话啊。", "expected": True},
 {"text": "说话啊，喂。", "expected": True},
 {"text": "哈喽哈喽，卡卡的了，你在哪里啊？", "expected": True},
 {"text": "喂喂，你在哪里？哈喽哈喽。", "expected": True},
 {"text": "什么耐心？", "expected": True},
 {"text": "不过背什么啊什么就断掉了。", "expected": True},
 {"text": "不过背什么啊，什么就断掉了？喂。", "expected": True},
 {"text": "哈喽，你在台北哪里？", "expected": True},
 {"text": "嘿，你住那里哦？你住新一区吗？还是在那边工作？", "expected": True},
 {"text": "哈喽哈喽，工作哦，那你住哪里？", "expected": True},
 {"text": "但是，喂，住在哪里？", "expected": True},
 {"text": "但是，喂，住在哪里？住在我心里。", "expected": True},
 {"text": "但是，喂，住在哪里？住在我心里，是不是？", "expected": True},
 {"text": "但是，喂，住在哪里？住在我心里，是不是？哈喽。", "expected": True},
 {"text": "啊，哈。", "expected": True},
 {"text": "没有啊。", "expected": True},
 {"text": "对呀对呀，说的对呀。", "expected": True},
 {"text": "好，你说你说，我听。", "expected": True},
 {"text": "好，你说你说，我听说啊。", "expected": True},
 {"text": "啊，怎么不说了？", "expected": True},
 {"text": "没关系没关系，最近怎么了？哈喽哈喽，快点讲话啊。", "expected": True},
 {"text": "嘿嘿，在准备什么？", "expected": True},
 {"text": "嘿嘿，在准备什么？喂。", "expected": True},
 {"text": "哦，嘿嘿，很好很好很好啊。", "expected": True},
 {"text": "我很好很好啊，嘿嘿，很好啊，不错不错，你开心我就开心。", "expected": True},
 {"text": "软了哦软了哦，我帮你把它揉得更软。", "expected": True},
 {"text": "给你捏啊给你捏啊，我也想捏你的啊。", "expected": True},
 {"text": "喂，哎。", "expected": True},
 {"text": "对呀，今天刚怎么了？", "expected": True},
 {"text": "嗯，哎，这样把口红涂一涂，然后再来亲亲我。", "expected": True},
 {"text": "可以吗？", "expected": True},
 {"text": "啊哈哈，我问一下啊。", "expected": True},
 {"text": "为什么你那边会卡卡的？", "expected": True},
 {"text": "啊，都快下雨了，这地上都湿湿的。", "expected": True},
 {"text": "对啊，拥抱在一起更好，对不对？", "expected": True},
 {"text": "那我这啊脸红了，脸红好啊，脸红对身体比较健康啊，气血循环比较好。", "expected": True},
 {"text": "你现在上班吗？", "expected": True},
 {"text": "啊你上班还，我偷偷打电话。", "expected": True},
 {"text": "对啊，打电话为了你啊。", "expected": True},
 {"text": "没关系，脸红好啊，总比脸白好，也比脸青脸", "expected": True},
 {"text": "啊讲话啊，要讲话啊。", "expected": True},
 {"text": "好啊好啊。", "expected": True},
 {"text": "啊，好好好，我想抱抱你，亲亲你。", "expected": True},
 {"text": "可以吗可以吗？", "expected": True},
 {"text": "你好。", "expected": True},
 {"text": "哦，抱一辈子。", "expected": True},
 {"text": "啊，大，当然大。", "expected": True},
 {"text": "没有，我在外。", "expected": True},
 {"text": "Hey", "expected": True},
 {"text": "可以吗？行不行？", "expected": True},
 {"text": "你要唱给我看吗？", "expected": True},
 {"text": "怎么说？呃，我我哦，你喜欢敢爱敢恨的哦。", "expected": True},
 {"text": "嗯嗯嗯，爱就好爱就好。", "expected": True},
 {"text": "好啊，我们怎样？说啊。", "expected": True},
 {"text": "对对，好啊好啊，有空的话都联系，可以吗？", "expected": True},
 {"text": "嗯，嗯，嗯，嗯。", "expected": True},
 {"text": "我现在不忙，你现在不忙吗？", "expected": True},
 {"text": "嗯哼，正好在整理东西哦。", "expected": True},
 {"text": "那很那很好啊。", "expected": True},
 {"text": "那好那很好啊，不错哦。", "expected": True},
 {"text": "心花怒放，其实你知道。", "expected": True},
 {"text": "其实，啊哈，你讲你讲。", "expected": True},
 {"text": "嘿嘿，拜拜啦，你这样说哦。", "expected": True},
 {"text": "好拜，好拜拜拜拜。", "expected": True},
 {"text": "拜拜。", "expected": True},
 {"text": "拜拜，好。", "expected": True},
 {"text": "不忙不忙，那你现在在做什么呢？", "expected": True},
 {"text": "今天过过得很好。", "expected": True},
 {"text": "放了很多糖啊。", "expected": True},
 {"text": "你声音真好听啊。", "expected": True},
 {"text": "真肉。", "expected": True},
 {"text": "快点，快去睡，睡觉了。", "expected": True},
 {"text": "听得到。", "expected": True},
 {"text": "听得到。", "expected": True},
 {"text": "这里不能养哟。", "expected": True},
 {"text": "那你分享照片呀。", "expected": True},
 {"text": "去分享你手机拍的照片啊。", "expected": True},
 {"text": "看你拍的所有照片，可以吗？", "expected": True},
 {"text": "可以啊。", "expected": True},
 {"text": "你是没有看到照片啊？", "expected": True},
 {"text": "What could we do? What could we do?", "expected": True},
 {"text": "妈妈最最最喜欢的是哪个？", "expected": True},
 {"text": "我都没接触过，是是过程呢。我买买上就试试了。", "expected": True},
 {"text": "我都没接触过，接触过什么呢？我妹妹上一次接触过。嗨，这个是什么字？嗨，我奶奶", "expected": True},
 {"text": "可以", "expected": True},
 {"text": "什么东西？", "expected": True},
 {"text": "嗯，鬼灭之刃", "expected": True},
 {"text": "在做什么呢？", "expected": True},
 {"text": "I'm OK.", "expected": True},
 {"text": "嗯，没有。", "expected": True},
 {"text": "喝好。", "expected": True},
 {"text": "我在卖回收呀，我我先我等，我要去，我现在去，要去准备马喽。", "expected": True},
 {"text": "换过来吧，换过来吧。", "expected": True},
 {"text": "我刚下班。", "expected": True},
 {"text": "你又按不到。", "expected": True},
 {"text": "你也捞不到啊。", "expected": True},
 {"text": "不出呢", "expected": True},
 {"text": "我说你住在哪里？", "expected": True},
 {"text": "我在早餐店工作。", "expected": True},
 {"text": "做早餐跟卖早餐不都一样吗？", "expected": True},
 {"text": "嗨，姐姐。", "expected": True},
 {"text": "我很好，我也，听到姐姐的声音我就很开心。", "expected": True},
 {"text": "你是 AI 吗？", "expected": True},
 {"text": "你确定你不是 AI 吗？", "expected": True},
 {"text": "你看", "expected": True},
 {"text": "能听到我说话吗？", "expected": True},
 {"text": "你发，没看到你发信息啊。", "expected": True},
 {"text": "太多信息了，你没有发给我啊。", "expected": True},
 {"text": "你几岁啊？", "expected": True},
 {"text": "嗯，你是弟弟啊，你比我", "expected": True},
 {"text": "你叫我几岁", "expected": True},
 {"text": "我才20岁。", "expected": True},
 {"text": "那你叫我宝贝吧。", "expected": True},
 {"text": "那你那你希望我叫你什么？", "expected": True},
 {"text": "可是你都叫我宝贝了，我怎么可能叫你小弟弟？", "expected": True},
 {"text": "那你希望我叫你什么？你想听什么称呼？", "expected": True},
 {"text": "你好呀。", "expected": True},
 {"text": "哈哈哈，你老公。哈哈哈。", "expected": True},
 {"text": "你在干嘛？", "expected": True},
 {"text": "你确定不叫你老公吗？你确定不要吗？你不会伤心吗？不会失望吗？", "expected": True},
 {"text": "可是我当成了老公。", "expected": True},
 {"text": "我都叫你老公了，我都叫你老公了，你都不叫我老婆吗？", "expected": True},
 {"text": "你在做什么？", "expected": True},
 {"text": "不忙不忙。", "expected": True},
 {"text": "嗯，no no no", "expected": True},
 {"text": "有。", "expected": True},
 {"text": "没有啊，老板一直叫我", "expected": True},
 {"text": "有有", "expected": True},
 {"text": "老板刚刚发消息给我讲，今天我不想讲。", "expected": True},
 {"text": "你在干什么呢？", "expected": True},
 {"text": "你在干嘛？没听清。", "expected": True},
 {"text": "你还在上学呀？你上几年级了？", "expected": True},
 {"text": "你学的什么专业呀？", "expected": True},
 {"text": "那我的心情很郁闷，我应该怎么治疗呀？", "expected": True},
 {"text": "被 CPU 了。", "expected": True},
 {"text": "被 PUA 了，被别人 PUA 了。", "expected": True},
 {"text": "可以呀，我中午吃饭都没吃下去。", "expected": True},
 {"text": "你帮我分析一下原因。", "expected": True},
 {"text": "也不是，就是我昨天看了一个笑话，但是我没有笑，所以我很郁闷。", "expected": True},
 {"text": "先忙了，拜拜。", "expected": True},
 {"text": "我不忙呀，你在干嘛呢？", "expected": True},
 {"text": "能听到啊。", "expected": True},
    ]

    print("\n测试结果:")

    positive_cases_count = 0
    positive_cases_right_count = 0
    negative_cases_count = 0
    negative_cases_right_count = 0
    failed_cases_count = 0
    for case in test_cases:
        resp = requests.post(
            f"{BASE_URL}/v1/text/eou",
            json={"text": case["text"], "threshold": 0.4},
        )
        result = resp.json()

        status = "✅" if result["is_end"] == case["expected"] else "❌"
    
        if case["expected"]:
            positive_cases_count += 1
            if result["is_end"] == case["expected"]:
                positive_cases_right_count += 1
        else:
            negative_cases_count += 1
            if result["is_end"] == case["expected"]:
                negative_cases_right_count += 1

        if result["is_end"] != case["expected"]:
            failed_cases_count += 1
        print(f"  {status} \"{case['text']}\"")
        print(
            f"      概率: {result['probability']:.4f}, 判断: {'说完了' if result['is_end'] else '没说完'}"
        )
    print(f"测试用例数量: {len(test_cases)}")
    print(f"正样本数量: {positive_cases_count}, 正样本正确数量: {positive_cases_right_count}, 正样本正确率: {positive_cases_right_count / positive_cases_count:.2%}")
    print(f"负样本数量: {negative_cases_count}, 负样本正确数量: {negative_cases_right_count}, 负样本正确率: {negative_cases_right_count / negative_cases_count:.2%}")
    print(f"失败用例数量: {failed_cases_count}")
    print(f"失败用例比例: {failed_cases_count / len(test_cases):.2%}")



if __name__ == "__main__":
    # 1. 测试 embed
    # emb = test_embed()

    # # 2. 测试 score
    # test_score(emb)

    # # 3. 测试 extract
    # test_extract(emb)

    # 4. 测试 EOU 句末检测
    test_eou()


    print("=" * 50)
    print("所有测试完成！")
