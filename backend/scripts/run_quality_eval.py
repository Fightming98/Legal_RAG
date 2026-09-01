"""质量评估：采集 ROUGE-L / 检索相关性 / 忠实度 三维数据"""
import asyncio, json, sys, time, os
# 将 backend 目录加入 sys.path，保证脚本以 `python -m scripts.run_quality_eval` 方式运行时能导入 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pipeline import RAGPipeline, PipelineConfig, QueryTransformStrategy, RerankStrategy, GenerationStrategy
from app.services.quality_service import evaluate_single_query, aggregate_quality
from app.models.schemas import SourceDocument

# 40 个用于质量评估的问题：其中前 8 条与 reference_answers.json 的参考答案对应（可算 ROUGE），
# 其余为真实咨询语料，用于评估检索相关性与忠实度。默认使用标准模式（none+simple+standard）
EVAL_QUERIES = [
    "民法典中关于合同成立的规定是什么？",
    "故意杀人罪的量刑标准是什么？",
    "劳动合同法中关于经济补偿金的规定",
    "交通事故损害赔偿的法律依据",
    "知识产权侵权的认定标准",
    "公司法中股东代表诉讼的条件",
    "行政诉讼的受案范围",
    "最高法关于民间借贷利率的指导案例",
    "你好，请问交通事故发生了伤着住院治疗需要赔付是怎么样流程？",
    "因再次酒驾而吊销驾驶证，有没有办法尽快拿到驾驶证。",
    "昨天驾驶无牌摩托车，交警队拉走了，驾驶证要扣多少分",
    "注册资金200万是什么意思，是指实际资产吗",
    "网上平台贷款受法律保护吗？？？",
    "借条一定要有保证人才算有法律效应么",
    "欠别人一万多别人按天算百分之0.35是不是放高利贷",
    "信用卡还不起了，银行都有哪些处罚，可以协商么？",
    "个人购买社保10，单位买5年后满50岁的女性可以办理退休吗",
    "私人老板请员工上班27个月后，员工自已辞职，9个月后又来上班，以前的工龄可以跟现在的一起算吗",
    "在成都XX公司工作，自动离职他不发工资怎么办？",
    "公司在员工没转证之前就购买保险合法吗",
    "商标法实施细则中对于商标专有使用权的解释",
    "注册商标后有人用同样名称从事文化宣传和商业经营活动也不向注册人告知是否属于侵权属于侵权？",
    "分享画家作品算侵权吗？我在堆糖网找的图片发表到XX讯这个软件上上可以吗",
    "在官网购买了5套（一个疗程）的祛斑产品，用第一套就发生过敏现象！（之前询问过客服我是敏感皮肤能否过敏可以么？",
    "申请复议是什么意思，申请复议了之后该怎么办？",
    "对消防处罚有异议，行政处罚说我们违反了第58条，其中的人员密集场所，未经消防安全检查擅自投入使用",
    "想写一份东莞市的卫计局行政复议申请书，不知道如何写",
    "有限责任公司股东变更都需要什么资料，我自己去工商局能办理吗？需要花钱吗大概流程是什么样的？",
    "我与朋友合伙，各出资30万，占比4,6分，我占4。 期间投入成本11万那么最后结账怎么算那么最后结账怎么算？",
    "母公司吸收合并子公司，原企业法律承续或分割情况的说明材料应如何填写",
    "父母无遗嘱遗产房姐要继承析产弟",
    "我父亲去世之前上午叫我找两个朋友做视频遗嘱，遗嘱说他把所有的遗产都留给他的儿子我没有说详细的遗产请问这个遗嘱有效吗？",
    "我想把我的房子过户给儿子，但儿子现在在国外留学，儿子不在可以过户吗？过户费咋算",
    "在嘉兴秀洲区盛世豪庭135.08平方的房产，4个人共同所有，每人25%的份额法院已经认定这样的房产能强制执行拍卖吗？",
    "民事诉讼的时效期是三年吗，过了三年还可以提起诉讼吗",
    "起诉别人打官司首先要做什么，然后做什么，我想知道有哪几步骤",
    "取保候审可以出国吗？就是在网上赌博做推广人员，然后被抓，去没有一个月现在想去国外旅游可以出去吗？",
    "因盗窃罪被拘留十天，后来派出所说取保候审，当时交了5000元取保金，受害人钱已退还是不是还要被判刑？",
    "玩黑彩输了150万都是骗来的，范了什么法",
    "身份证过期了在北京能办临时身份证吗",
]

async def main():
    # 连通性验证：触发一次远端 API 调用，确认硅基流动密钥/模型可用
    print("=== Verifying model connectivity ===")
    from app.core.llm import get_llm
    from app.core.embeddings import get_embeddings
    await get_llm().ainvoke("你好")
    get_embeddings().embed_query("测试")
    print("=== Connectivity check done ===\n")

    all_quality = []
    for i, q in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {q[:30]}...", end=" ", flush=True)
        t0 = time.time()
        # 固定使用标准模式（不重写、简单重排、标准生成）作为质量评估基线
        cfg = PipelineConfig(
            query_transform=QueryTransformStrategy.NONE,
            rerank_strategy=RerankStrategy.SIMPLE,
            generation_strategy=GenerationStrategy.STANDARD,
            use_kg=False, top_k=5,
            collection_names=["laws", "cases"],
        )
        pipe = RAGPipeline(cfg)
        resp = await pipe.execute(q)

        # 将管线返回的来源文档转为 SourceDocument 模型，供质量评估函数消费
        sources_for_eval = [
            SourceDocument(content=s.content, metadata=s.metadata, score=s.score)
            for s in resp.sources
        ]
        # 对单条查询计算 ROUGE（有参考答案时）、检索相关性、忠实度三项指标
        qm = await evaluate_single_query(q, resp.answer, sources_for_eval)
        all_quality.append(qm)
        elapsed = time.time() - t0

        # 打印单条评估结果（无参考答案时 ROUGE 显示 no-ref）
        rouge_str = f"R1={qm.rouge.rouge_1:.3f} R2={qm.rouge.rouge_2:.3f} RL={qm.rouge.rouge_l:.3f}" if qm.rouge else "no-ref"
        rel_str = f"rel={qm.retrieval_relevance.avg_relevance:.1f}" if qm.retrieval_relevance else "N/A"
        faith_str = f"faith={qm.faithfulness.score:.1f}" if qm.faithfulness else "N/A"
        print(f"{elapsed:.0f}s | {rouge_str} | {rel_str} | {faith_str}")

    # 汇总全部查询的均值指标
    agg = aggregate_quality(all_quality)
    print("\n" + "="*50)
    print(f"ROUGE-1:      {agg.avg_rouge_1:.4f}")
    print(f"ROUGE-2:      {agg.avg_rouge_2:.4f}")
    print(f"ROUGE-L:      {agg.avg_rouge_l:.4f}")
    print(f"检索相关性:   {agg.avg_retrieval_relevance:.2f} / 10")
    print(f"忠实度:       {agg.avg_faithfulness:.2f} / 10")
    print(f"评估条数:     {agg.evaluated_count}")

    # 组装明细数据：对可能为 None 的指标做判空，避免 JSON 序列化报错
    details = []
    for qm in all_quality:
        details.append({
            "query": qm.query,
            "rouge": {"r1": qm.rouge.rouge_1, "r2": qm.rouge.rouge_2, "rl": qm.rouge.rouge_l} if qm.rouge else None,
            "retrieval_relevance": qm.retrieval_relevance.avg_relevance if qm.retrieval_relevance else None,
            "relevance_ratio": qm.retrieval_relevance.relevance_ratio if qm.retrieval_relevance else None,
            "faithfulness": qm.faithfulness.score if qm.faithfulness else None,
            "faithfulness_expl": qm.faithfulness.explanation if qm.faithfulness else None,
        })
    # 将汇总与明细落盘为 JSON，供后续分析
    with open("/tmp/quality_eval_results.json", "w") as f:
        json.dump({"aggregated": {"rouge_1": agg.avg_rouge_1, "rouge_2": agg.avg_rouge_2, "rouge_l": agg.avg_rouge_l,
                                    "retrieval_relevance": agg.avg_retrieval_relevance, "faithfulness": agg.avg_faithfulness},
                    "details": details}, f, ensure_ascii=False, indent=2)
    print("\nSaved to /tmp/quality_eval_results.json")

asyncio.run(main())
