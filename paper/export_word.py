"""Export the paper through Pandoc with editable tables and Office equations."""

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree


def prepare_latex(source):
    counters = Counter()
    stack = []
    refs = {}
    section = 0
    subsection = 0
    appendix = False
    current_heading = ""
    events = re.compile(
        r"\\(?P<env>begin|end)\{(?P<kind>table\*?|figure\*?|equation)\}"
        r"|\\(?P<heading>section|subsection)\{(?P<title>[^{}]*)\}"
        r"|\\label\{(?P<label>[^{}]+)\}"
        r"|(?P<appendix>\\appendix\b)"
    )

    def event(match):
        nonlocal section, subsection, appendix, current_heading
        if match["appendix"]:
            appendix = True
            section = 0
            return ""
        if match["env"]:
            kind = match["kind"].rstrip("*")
            if match["env"] == "begin":
                counters[kind] += 1
                stack.append((kind, counters[kind]))
            else:
                if not stack or stack.pop()[0] != kind:
                    raise ValueError("Mismatched float/equation environment")
            return match[0]
        if match["heading"]:
            if match["heading"] == "section":
                section += 1
                subsection = 0
                current_heading = chr(64 + section) if appendix else str(section)
                display = ("附录 " if appendix else "") + current_heading
            else:
                subsection += 1
                prefix = chr(64 + section) if appendix else str(section)
                current_heading = f"{prefix}.{subsection}"
                display = current_heading
            return "\\" + match["heading"] + "{" + display + " " + match["title"] + "}"
        if match["label"]:
            if stack:
                refs[match["label"]] = str(stack[-1][1])
            else:
                refs[match["label"]] = current_heading
            return ""
        raise AssertionError(match[0])

    text = events.sub(event, source)
    if stack:
        raise ValueError("Unclosed environment")
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: refs[m[1]], text)
    text = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + refs[m[1]] + ")", text)

    # Let Pandoc expand resizebox as an identity macro; otherwise tables vanish.
    text = text.replace(
        r"\begin{document}",
        "\\renewcommand{\\resizebox}[3]{#3}\n\\begin{document}",
        1,
    )
    for kind, name in (("table", "表"), ("figure", "图")):
        count = [0]

        def caption(match):
            count[0] += 1
            return match[0].replace(
                r"\caption{", "\\caption{" + f"{name} {count[0]}　", 1
            )

        pattern = r"\\begin\{" + kind + r"\*?\}.*?\\end\{" + kind + r"\*?\}"
        text = re.sub(pattern, caption, text, flags=re.S)

    eq_number = [0]

    def equation(match):
        eq_number[0] += 1
        return "\\[" + match[1].strip() + "\\qquad (" + str(eq_number[0]) + ")\\]"

    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", equation, text, flags=re.S
    )
    bib_pattern = r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}"
    bibkeys = re.findall(bib_pattern, text)
    bibliography = {key: index + 1 for index, key in enumerate(bibkeys)}
    text = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda m: "[" + ", ".join(str(bibliography[k.strip()]) for k in m[1].split(",")) + "]",
        text,
    )
    text = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", r"\\section*{参考文献}", text)
    text = text.replace(r"\end{thebibliography}", "")
    text = re.sub(bib_pattern, lambda m: "\n\n\\noindent [" + str(bibliography[m[1]]) + "] ", text)
    return text, counters, len(bibkeys)


def node_counts(tree):
    counts = Counter()

    def walk(node):
        if isinstance(node, dict):
            if "t" in node:
                counts[node["t"]] += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return counts


def style_document(path):
    doc = Document(path)
    # Pandoc places adjacent figures in a layout table; stack them for legibility.
    for table in list(doc.tables):
        if any(cell.text.strip() for row in table.rows for cell in row.cells):
            continue
        drawings = table._tbl.xpath(".//w:p[w:r/w:drawing]")
        if not drawings:
            continue
        parent = table._tbl.getparent()
        index = parent.index(table._tbl)
        for paragraph in drawings:
            parent.insert(index, deepcopy(paragraph))
            index += 1
        parent.remove(table._tbl)
    for paragraph in doc.paragraphs:
        if paragraph.text == "Abstract":
            paragraph.text = "摘要"
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2)
        section.left_margin = section.right_margin = Cm(2)

    for style in doc.styles:
        if style.type not in (1, 2):
            continue
        style.font.name = "Times New Roman"
        props = style.element.get_or_add_rPr()
        fonts = props.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            props.insert(0, fonts)
        fonts.set(qn("w:eastAsia"), "宋体")
        style.font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.2
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 20), ("Heading 1", 16), ("Heading 2", 13)):
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.element.rPr.find(qn("w:rFonts")).set(qn("w:eastAsia"), "黑体")
        style.paragraph_format.keep_with_next = True

    for table in doc.tables:
        if not any(cell.text.strip() for row in table.rows for cell in row.cells):
            continue
        table.autofit = True
        if "Table Grid" in doc.styles:
            table.style = "Table Grid"
        # Prefer editable full-width tables over tiny conference-column text.
        width = table._tbl.tblPr.find(qn("w:tblW"))
        if width is not None:
            width.set(qn("w:type"), "pct")
            width.set(qn("w:w"), "5000")
        for index, row in enumerate(table.rows):
            no_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(no_split)
            if index == 0:
                header = OxmlElement("w:tblHeader")
                row._tr.get_or_add_trPr().append(header)
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(3)
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.paragraph_format.keep_with_next = index < len(table.rows) - 1
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        if index == 0:
                            run.bold = True
    for shape in doc.inline_shapes:
        shape.height = int(shape.height * Cm(17) / shape.width)
        shape.width = Cm(17)
    for paragraph in doc.paragraphs:
        if paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = 1
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.left_indent = Cm(0)
            paragraph.paragraph_format.right_indent = Cm(0)
            paragraph.paragraph_format.first_line_indent = Cm(0)
        if paragraph.text.startswith("表 "):
            paragraph.paragraph_format.keep_with_next = True
    doc.core_properties.title = "学习何时思考：面向大语言模型高效推理的自适应反思策略学习"
    doc.core_properties.subject = "可编辑论文初稿"
    doc.core_properties.author = "匿名作者"
    doc.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandoc", required=True)
    parser.add_argument("--source", default=str(Path(__file__).with_name("main.tex")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("adaptive_reflection_draft.docx")))
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    original = source.read_text(encoding="utf-8")
    latex, expected, references = prepare_latex(original)
    with tempfile.TemporaryDirectory(prefix="reflection_word_") as temp:
        prepared = Path(temp) / "word_source.tex"
        ast_path = Path(temp) / "document.json"
        prepared.write_text(latex, encoding="utf-8")
        common = [args.pandoc, str(prepared), "--from=latex", "--standalone"]
        subprocess.run(common + ["--to=json", "--output=" + str(ast_path)], check=True, cwd=source.parent)
        tree = json.loads(ast_path.read_text(encoding="utf-8"))
        counts = node_counts(tree)
        if counts["Table"] != expected["table"]:
            raise RuntimeError(f"Table loss: {counts['Table']} != {expected['table']}")
        if counts["DisplayMath"] != expected["equation"]:
            raise RuntimeError("Display equation count mismatch")
        expected_images = len(re.findall(r"\\includegraphics\b", original))
        if counts["Image"] != expected_images:
            raise RuntimeError("Image count mismatch")
        result = subprocess.run(
            common + ["--to=docx", "--resource-path=" + str(source.parent), "--output=" + str(output)],
            cwd=source.parent, text=True, capture_output=True, check=True,
        )
        if result.stderr.strip():
            raise RuntimeError("Pandoc warning requires inspection: " + result.stderr)
    style_document(output)
    with ZipFile(output) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
              "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
        all_tables = root.findall(".//w:tbl", ns)
        word_tables = sum(
            any((node.text or "").strip() for node in table.findall(".//w:t", ns))
            for table in all_tables
        )
        office_math = len(root.findall(".//m:oMath", ns))
        text = "".join(root.itertext())
        if word_tables != expected["table"]:
            raise RuntimeError("Word table count mismatch")
        if office_math < counts["Math"]:
            raise RuntimeError("An equation was not converted to native Office Math")
        if any(token in text for token in (r"\label{", r"\ref{", r"\begin{", r"\cite{")):
            raise RuntimeError("Unconverted LaTeX remains")
        report = {
            "source": str(source), "output": str(output), "tables": word_tables,
            "figure_layout_tables": len(all_tables) - word_tables,
            "images": counts["Image"], "display_equations": expected["equation"],
            "native_office_math": office_math, "bibliography_entries": references,
            "size_bytes": output.stat().st_size,
        }
    output.with_suffix(".conversion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
