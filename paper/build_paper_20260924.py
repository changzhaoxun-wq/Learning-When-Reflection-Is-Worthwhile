"""Export the dated manuscript and evidence-bound supplement; no model calls."""
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
import argparse
import hashlib
import json
import re
import subprocess

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STEM = "HIERARCHICAL_REFLECTION_PAPER_20260924"
MAIN_JSON = ROOT / "outputs/math_full_test_failure_accounted_v1/evaluation/summary.json"
TRANSFER = ROOT / "outputs/math_transfer_locked_v1/evaluation"
SOURCES = [
    MAIN_JSON,
    ROOT / "src/math_gate_ranking.py",
    ROOT / "src/math_gate_cpu_ablation.py",
    ROOT / "src/train_fixed_math_policy.py",
    ROOT / "outputs/math_gate_ranking_20260916_seed42/protocol.json",
    ROOT / "outputs/math_full_test_v1/policy_lock.json",
    PAPER / "results_audit_20260918/attribution.json",
    ROOT / "outputs/math_gate_hparam_v1/REPORT.md",
    ROOT / "outputs/math_transfer_locked_v1/generation_protocol.json",
    ROOT / "outputs/math_transfer_locked_v1/policy_lock.json",
] + [TRANSFER / name / "full.json" for name in ("gsm1k", "svamp", "multiarith")] + [
    ROOT / "src/evaluate_accounted_math_test.py",
    ROOT / "src/analyze_gate_random_baseline.py",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def label(group):
    loss = "CE" if group["recipe"] == "weighted_ce_control" else "CE+Rank"
    branch = "Gate-Round" if group["branch"] == "original_round" else "Gate-R1"
    return f"{loss}/{branch}"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |",
         "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def spread(metric, scale=1):
    return f"{scale * metric['mean']:.2f} ± {scale * metric['std']:.2f}"


def supplement(data):
    lines = [
        "---", 'title: "分层数学反思路由：补充材料"',
        'subtitle: "2026-09-24，依据保存的结构化结果生成"', "lang: zh-CN", "---", "",
        "本材料不是新增实验。全部指标复用已保存轨迹，不调用模型、不更新策略。",
        "准确率和反思率以百分数报告；准确率标准差和差值区间以百分点报告。",
        "人工科学复核尚待实际作者完成。", "",
        "## A 逐种子检查点与测试结果", "",
        "Gate 的 epoch 与阈值由验证规则确定；固定 Round 为 epoch 1。"
        "各模型文件及哈希保存在原始锁定清单，未按测试成绩挑选种子。", "",
    ]
    rows = []
    for p in data["policies"]:
        m = p["metrics"]
        rows.append([label(p), p["seed"], f"{p['validation_call_budget']:.0%}",
                     p["epoch"], f"{p['direct_threshold']:.8f}",
                     f"{100*m['accuracy']:.2f}", f"{m['avg_token_cost']:.2f}",
                     f"{100*m['reflection_rate']:.2f}", m["execution_failure_count"]])
    lines += [table(["配置", "Seed", "验证预算", "Epoch", "Direct阈值",
                     "Acc.(%)", "总token", "反思率(%)", "失败"], rows), "",
              "## B 主实验输入输出拆分与配对差值", ""]
    rows = []
    for g in data["groups"]:
        m = g["metrics"]
        rows.append([label(g), f"{g['validation_call_budget']:.0%}",
                     spread(m["avg_prompt_tokens"]), spread(m["avg_completion_tokens"]),
                     spread(m["avg_token_cost"]), spread(m["fixes"]),
                     spread(m["harms"])])
    lines += [table(["配置", "验证预算", "输入token", "输出token", "总token",
                     "修复题数", "引错题数"], rows), "",
              "输入输出均含失败尝试；各项分别四舍五入，显示值之和可能差 0.01。", ""]
    rows = []
    for g in data["groups"]:
        for base in ("direct", "fixed_reflect_1", "independent"):
            a = g["paired_bootstrap"][base]["accuracy"]
            t = g["paired_bootstrap"][base]["avg_token_cost"]
            rows.append([label(g), f"{g['validation_call_budget']:.0%}", base,
                         f"{100*a['observed_difference']:+.2f} "
                         f"[{100*a['ci95_lower']:+.2f}, {100*a['ci95_upper']:+.2f}]",
                         f"{t['observed_difference']:+.2f} "
                         f"[{t['ci95_lower']:+.2f}, {t['ci95_upper']:+.2f}]"])
    lines += [table(["配置", "预算", "比较基线", "Acc.差[95%CI], pp",
                     "token差[95%CI]"], rows), "",
              "Bootstrap 先按题平均种子，再按题配对重采样；区间不是同时置信区间。", "",
              "## C 匹配调用数的随机对照", "",
              "均匀随机保留全量题目的冻结 Round 预测；轮次分层随机在完整的"
              "预测轮次池内匹配各轮次数量。均不匹配总 token。", ""]
    rows = []
    for g in data["groups"]:
        for control in ("uniform_gate", "round_stratified_gate"):
            r = g["random_expected"][control]
            ci = g["paired_bootstrap"][control]["accuracy"]
            rows.append([label(g), f"{g['validation_call_budget']:.0%}",
                         "均匀" if control == "uniform_gate" else "轮次分层",
                         f"{100*r['accuracy']['mean']:.2f}",
                         f"{r['avg_token_cost']['mean']:.2f}",
                         f"{100*ci['observed_difference']:+.2f} "
                         f"[{100*ci['ci95_lower']:+.2f}, {100*ci['ci95_upper']:+.2f}]"])
    lines += [table(["配置", "预算", "随机方式", "随机Acc.(%)",
                     "随机token", "策略Acc.差[95%CI], pp"], rows), "",
              "## D 全部冻结迁移配置", "",
              "不使用目标数据选择 epoch、阈值或种子。GSM1K 独立强模型成本存在"
              "缺失 usage，以下基线表明确标记为下界；不报告尚需复核的迁移随机对照。", ""]
    for dataset in ("gsm1k", "svamp", "multiarith"):
        d = read(TRANSFER / dataset / "full.json")
        lines += [f"### {dataset.upper()}，n={d['n']}", ""]
        rows = []
        for name, m in d["baselines"].items():
            lower = "≥" if not m.get("token_cost_complete", True) else ""
            rows.append([name, f"{100*m['accuracy']:.2f}",
                         lower + f"{m['avg_token_cost']:.2f}",
                         f"{m['fixes']}/{m['harms']}", m["execution_failure_count"]])
        lines += [table(["基线", "Acc.(%)", "已知总token", "修复/引错", "失败"], rows), ""]
        rows = []
        for g in d["groups"]:
            m = g["metrics"]
            rows.append([label(g), f"{g['validation_call_budget']:.0%}",
                         spread(m["accuracy"], 100), spread(m["avg_prompt_tokens"]),
                         spread(m["avg_completion_tokens"]),
                         spread(m["avg_token_cost"]),
                         f"{100*m['reflection_rate']['mean']:.2f}"])
        lines += [table(["配置", "预算", "Acc.(%)", "输入token",
                         "输出token", "总token", "反思率(%)"], rows), ""]
    lines += ["## E 训练曲线与选择解释", "",
              "每个 Gate 训练 10 个 epoch，检查点由最终验证答对数及成本选取，而非"
              "最低训练 loss 或最后一个 epoch。上文逐点列出所选 epoch，避免用单一"
              "epoch 概括所有配置。以下 seed 42 仅作为训练诊断示例，不是按测试最好种子挑选。",
              "普通 CE 诊断曲线不等于加权 CE 加排序的优化目标；准确率曲线如标注"
              "classification，表示动作分类，不等于最终答案准确率。", ""]
    for recipe, name in (("weighted_ce_control", "CE"),
                         ("weighted_ce_pairwise", "CE+Rank")):
        directory = ROOT / f"outputs/math_gate_ranking_20260916_seed42/{recipe}"
        lines += [f"### {name}，seed 42", "",
                  f"![{name} 的已保存训练诊断曲线。](../{directory.relative_to(ROOT)}/training_curve.png)", "",
                  f"历史文件：{directory.relative_to(ROOT)}/history.json。", ""]
    lines += ["## F 证据来源", "",
              "方法与数值的来源哈希、机器核对范围及尚待人工核验项目见"
              " PAPER_ASSEMBLY_20260924.json 和 PAPER_ASSEMBLY_NOTES_20260924.md。"
              "本材料不声称已通过期刊审查或完成独立确认。", ""]
    return "\n".join(lines)


def check_manuscript(data, text):
    # Check table rows against raw values, rather than finding unanchored numbers.
    aliases = {"direct": "Direct", "fixed_reflect_1": "Fixed R1",
               "fixed_reflect_2": "Fixed R2", "independent": "Strong independent",
               "oracle_accuracy_first": "D/R1/R2 Oracle"}
    for name, m in data["baselines"].items():
        row = (f"| {aliases[name]} | {100*m['accuracy']:.2f} | "
               f"{m['avg_prompt_tokens']:.2f} | {m['avg_completion_tokens']:.2f} | "
               f"{m['avg_token_cost']:.2f} | {m['fixes']}/{m['harms']} | "
               f"{m['execution_failure_count']} |")
        assert row in text, row
    for g in data["groups"]:
        loss, branch = label(g).split("/")
        m = g["metrics"]
        row = (f"| {loss} | {branch} | {g['validation_call_budget']:.0%} | "
               f"{spread(m['accuracy'],100)} | {spread(m['avg_token_cost'])} | "
               f"{100*m['reflection_rate']['mean']:.2f} |")
        assert row in text, row
    assert len(data["policies"]) == 36 and data["n"] == 1319
    for dataset in ("gsm1k", "svamp", "multiarith"):
        transfer = read(TRANSFER / dataset / "full.json")
        group = next(g for g in transfer["groups"]
                     if g["recipe"] == "weighted_ce_control"
                     and g["branch"] == "original_round"
                     and g["validation_call_budget"] == .1)
        m = group["metrics"]
        suffix = (f"| {spread(m['accuracy'], 100)} | "
                  f"{spread(m['avg_token_cost'])} | "
                  f"{100*m['reflection_rate']['mean']:.2f} |")
        assert suffix in text, (dataset, suffix)
    for bad in ("91.73", "[claim:", "[evidence:", "@@"):
        assert bad not in text, bad
    attrib = read(PAPER / "results_audit_20260918/attribution.json")["per_policy"]
    for a in attrib:
        assert a["gate_missed_repair"] + a["round_missed_repair"] + a["introduced_error"] == a["oracle_correct"] - a["correct"]
    return {"baseline_rows": 5, "policy_group_rows": 12, "transfer_group_rows": 3,
            "policies": 36,
            "attribution_identities": len(attrib), "numeric_check": "passed",
            "scope": "Selected tables and algebraic consistency; not full scientific verification"}


def style(path, title):
    doc = Document(path)
    is_main = path.stem == STEM
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2)
        section.left_margin = section.right_margin = Cm(2)
    for s in doc.styles:
        if s.type not in (1, 2):
            continue
        s.font.name = "Times New Roman"
        s.font.color.rgb = RGBColor(0, 0, 0)
        fonts = s.element.get_or_add_rPr().find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            s.element.rPr.append(fonts)
        fonts.set(qn("w:eastAsia"), "Noto Serif CJK SC")
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(5)
    for name, size in (("Title", 19), ("Subtitle", 12),
                       ("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)):
        s = doc.styles[name]
        s.font.size = Pt(size)
        s.paragraph_format.keep_with_next = True
    if is_main:
        for p in doc.paragraphs:
            if p.style.name == "Title" and "：" in p.text:
                p.text = p.text.replace("：", "：\n", 1)
    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = 1
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
    for t in doc.tables:
        t.autofit = False
        count = len(t.columns)
        if count == 7:
            widths = [3.8, 2, 2.2, 2.2, 2.2, 2.2, 2.4]
        elif count == 6:
            widths = [2.0, 2.8, 1.8, 3.5, 3.8, 3.1]
        elif count == 4:
            widths = [3.3, 4.0, 3.3, 6.4]
        elif count == 5:
            widths = [3.0, 4.8, 3.7, 3.1, 1.4]
        else:
            widths = [17 / count] * count
        # Long supplementary identifiers need a wider first column.
        if not is_main:
            widths = [3.7] + [(17 - 3.7) / (count - 1)] * (count - 1)
        for col, width in zip(t.columns, widths):
            col.width = Cm(width)
        if "Table Grid" in doc.styles:
            t.style = "Table Grid"
        for i, row in enumerate(t.rows):
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement("w:cantSplit"))
            if i == 0:
                props.append(OxmlElement("w:tblHeader"))
            for cell, width in zip(row.cells, widths):
                cell.width = Cm(width)
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = i == 0
                    p.paragraph_format.line_spacing = 1
                    p.paragraph_format.space_after = Pt(3)
                    for r in p.runs:
                        r.font.size = Pt(9)
                        if i == 0:
                            r.bold = True
    for shape in doc.inline_shapes:
        if shape.width > Cm(17):
            shape.height = int(shape.height * Cm(17) / shape.width)
            shape.width = Cm(17)
    doc.core_properties.title = title
    doc.core_properties.author = ""
    doc.core_properties.subject = "作者审阅稿；尚非投稿定稿"
    doc.save(path)


def export(pandoc, source):
    ast = json.loads(subprocess.check_output(
        [pandoc, str(source), "-f", "markdown+tex_math_dollars", "-t", "json"], text=True))
    counts = Counter()
    def walk(obj):
        if isinstance(obj, dict):
            if "t" in obj:
                counts[obj["t"]] += 1
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(ast)
    output = source.with_suffix(".docx")
    r = subprocess.run([pandoc, str(source), "-f", "markdown+tex_math_dollars",
                        "-t", "docx", "--standalone", "--resource-path", str(PAPER),
                        "-o", str(output)], capture_output=True, text=True, check=True)
    # This Pandoc build lacks Chinese label translations; our captions are explicit.
    allowed = re.compile(
        r"(?:\[WARNING\] Could not load translations for zh-CN|"
        r"data file translations/zh.yaml not found|"
        r"\[WARNING\] The term (?:Abstract|Table|Figure) has no translation defined\.)"
    )
    warnings = [line.strip() for line in r.stderr.splitlines() if line.strip()]
    if any(not allowed.fullmatch(line) for line in warnings):
        raise RuntimeError(r.stderr)
    style(output, "分层反思路由论文" if source.stem == STEM else "分层反思路由补充材料")
    with ZipFile(output) as z:
        xml = etree.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    got = {"tables": len(xml.findall(".//w:tbl", ns)),
           "native_equations": len(xml.findall(".//m:oMath", ns)),
           "drawings": len(xml.findall(".//w:drawing", ns))}
    assert got["tables"] == counts["Table"], (got, counts)
    assert got["native_equations"] == counts["Math"], (got, counts)
    assert got["drawings"] == counts["Image"], (got, counts)
    return {"source": source.name, "output": output.name, **got, "warnings": warnings,
            "size_bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandoc", required=True)
    args = parser.parse_args()
    source = PAPER / f"{STEM}.md"
    data = read(MAIN_JSON)
    checks = check_manuscript(data, source.read_text(encoding="utf-8"))
    extra = PAPER / "HIERARCHICAL_REFLECTION_SUPPLEMENT_20260924.md"
    extra.write_text(supplement(data), encoding="utf-8")
    exports = [export(args.pandoc, p) for p in (source, extra)]
    manifest = {
        "date": "2026-09-24", "document": source.name, "submission_ready": False,
        "human_verification": "pending", "machine_checks": checks,
        "exports": exports,
        "scope": "Mathematics only; no experiment code changes or LLM calls",
        "sources": [{"id": f"E{i:02d}", "path": str(p.relative_to(ROOT)),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "human_verified": False} for i, p in enumerate(SOURCES, 1)],
        "claim_groups": {
            "abstract_and_gsm8k_results": ["E01"],
            "architecture_and_training": ["E02", "E03", "E04", "E05", "E06"],
            "error_attribution": ["E07"],
            "validation_only_hparam": ["E08"],
            "transfer_protocol_and_results": ["E09", "E10", "E11", "E12", "E13"],
        },
        "public_reference_urls": re.findall(r"\[原文\]\((https://[^)]+)\)",
                                            source.read_text(encoding="utf-8")),
        "reference_review": "Primary publisher/arXiv pages opened by assistant on 2026-09-24; human verification pending",
    }
    (PAPER / "PAPER_ASSEMBLY_20260924.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "exports": exports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
