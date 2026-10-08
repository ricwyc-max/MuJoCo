"""把 REPORT.md 转成论文排版标准的 .docx。

    .venv/Scripts/python.exe scripts/make_report_docx.py

排版参照中文学位论文的通行规范：
- 封面：报告标题 + 学号/姓名/班级
- 正文：宋体小四（12pt）、西文 Times New Roman、1.5 倍行距、首行缩进 2 字符
- 标题：黑体，一级二号/二级小二/三级四号
- 表格：三线表（顶线、表头线、底线），宋体五号
- 图：居中，图题在图下方（宋体五号居中）
- 公式/代码：Consolas + 宋体，浅灰底纹，无首行缩进
- 目录：Word 域，打开文档时自动更新

只依赖 python-docx，不引入 pandoc 等外部工具，保证在别的机器上也能一键复现。
"""

from __future__ import annotations

import pathlib
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_MD = PROJECT_ROOT / "REPORT.md"
OUTPUT_DOCX = PROJECT_ROOT / "MuJoCo课程作业报告.docx"

# ---------------------------------------------------------------- 个人信息
STUDENT = dict(student_id="2304120130", name="吴尧承", class_name="智建一班")
COURSE_TITLE = "课程作业报告"
DOC_TITLE = "基于 MuJoCo 的机器人控制入门"
DOC_SUBTITLE = "——任务二：重力补偿与保持"

# ---------------------------------------------------------------- 排版参数
FONT_HEI = "黑体"  # 标题中文字体
FONT_SONG = "宋体"  # 正文中文字体
FONT_LATIN = "Times New Roman"  # 西文字体
FONT_MONO = "Consolas"

BODY_SIZE = Pt(12)  # 小四
BODY_LINE_SPACING = 1.5
FIRST_LINE_INDENT = Pt(24)  # 2 字符
TABLE_SIZE = Pt(10.5)  # 五号
CAPTION_SIZE = Pt(10.5)
CODE_SIZE = Pt(10)

IMAGE_WIDTH = Cm(14.5)


# ---------------------------------------------------------------- 字体工具


def set_run_font(run, *, east_asia=FONT_SONG, latin=FONT_LATIN, size=BODY_SIZE,
                 bold=False, italic=False, color=None):
    """同时设置西文字体与中文字体（python-docx 只设 ascii 不够，要显式设 eastAsia）。"""
    run.font.name = latin
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), east_asia)
    return run


def _shade(element, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    element.append(shd)


def shade_paragraph(paragraph, fill="F2F2F2"):
    _shade(paragraph._p.get_or_add_pPr(), fill)


# ---------------------------------------------------------------- 文档骨架


def setup_document() -> Document:
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_LATIN
    normal.font.size = BODY_SIZE
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_SONG)
    # 正文行距由各段落显式设置；这里保持单倍，否则 Word 自动生成的目录条目
    # 会继承 1.5 倍行距，24 条目录就撑满一页、溢到第二页去。
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(0)

    # 打开文档时自动更新域（目录页码才会正确）
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    doc.settings.element.append(update)

    return doc


def add_page_number_footer(section) -> None:
    """页脚居中显示"第 X 页"（只加在正文节，封面与目录不显示）。"""
    section.footer.is_linked_to_previous = False
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, size=Pt(10.5))

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    run._element.append(fld_begin)
    run._element.append(instr)
    run._element.append(fld_end)

    set_run_font(paragraph.add_run(" 页"), size=Pt(10.5))


def restart_page_numbering(section, start: int = 1) -> None:
    """让本节的页码重新从 1 开始（学术惯例：封面/目录不计入正文页码）。"""
    pg_num_type = OxmlElement("w:pgNumType")
    pg_num_type.set(qn("w:start"), str(start))
    section._sectPr.append(pg_num_type)


def add_cover(doc: Document, title: str) -> None:
    for _ in range(4):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run(COURSE_TITLE), east_asia=FONT_HEI, size=Pt(22), bold=False)

    for _ in range(2):
        doc.add_paragraph()

    # markdown 标题形如「主标题——副标题」，拆成两行排版更接近论文封面
    main, _, sub = title.partition("——")
    lines = [(main.strip(), Pt(18))]
    if sub.strip():
        lines.append((sub.strip(), Pt(16)))
    for text, size in lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(p.add_run(text), east_asia=FONT_HEI, size=size)

    for _ in range(6):
        doc.add_paragraph()

    for label, value in (("学    号", STUDENT["student_id"]),
                         ("姓    名", STUDENT["name"]),
                         ("班    级", STUDENT["class_name"])):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(p.add_run(f"{label}：{value}"), east_asia=FONT_HEI, size=Pt(14))

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_toc(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("目　　录"), east_asia=FONT_HEI, size=Pt(16))

    toc = doc.add_paragraph()
    run = toc.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "（在 Word 中按 Ctrl+A 后按 F9 更新目录）"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instr, separate, placeholder, end):
        run._element.append(element)
    # 目录后不额外插分页符：正文用"下一页分节"起头，多插一页会留空白页


# ---------------------------------------------------------------- 段落构件


def strip_inline(text: str) -> str:
    """去掉行内 markdown 标记，只留纯文本（用于标题、题注等不适合富文本的地方）。"""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"`([^`]+)`", r"\1", text)


def add_heading(doc: Document, text: str, level: int) -> None:
    """level: 1=一级(##) 2=二级(###) 3=三级(####)。"""
    sizes = {1: Pt(15), 2: Pt(13.5), 3: Pt(12)}
    spacing_before = {1: Pt(16), 2: Pt(12), 3: Pt(10)}
    spacing_after = {1: Pt(10), 2: Pt(8), 3: Pt(6)}

    heading = doc.add_heading(level=min(level, 3))
    heading.paragraph_format.space_before = spacing_before[level]
    heading.paragraph_format.space_after = spacing_after[level]
    heading.paragraph_format.line_spacing = 1.25
    # Word 内置标题样式是蓝色 Calibri Light，论文里要改成黑色黑体
    set_run_font(heading.add_run(strip_inline(text)), east_asia=FONT_HEI,
                 size=sizes[level], bold=(level >= 3), color=RGBColor(0, 0, 0))


def add_body_paragraph(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.first_line_indent = FIRST_LINE_INDENT
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = BODY_LINE_SPACING
    fmt.space_after = Pt(0)
    _add_inline(p, text)


def add_list_item(doc: Document, text: str, ordered: bool) -> None:
    p = doc.add_paragraph(style="List Number" if ordered else "List Bullet")
    fmt = p.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = BODY_LINE_SPACING
    fmt.space_after = Pt(0)
    fmt.left_indent = Pt(24)
    _add_inline(p, text)


def add_quote(doc: Document, lines: list[str]) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.left_indent = Pt(24)
    fmt.right_indent = Pt(12)
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = 1.4
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(6)
    _shade(p._p.get_or_add_pPr(), "F5F5F5")
    _add_inline(p, "".join(lines))


def add_code_block(doc: Document, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        fmt = p.paragraph_format
        fmt.left_indent = Pt(21)
        fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        fmt.line_spacing = 1.15
        fmt.space_before = Pt(6) if i == 0 else Pt(0)
        fmt.space_after = Pt(6) if i == len(lines) - 1 else Pt(0)
        shade_paragraph(p, "F2F2F2")
        set_run_font(p.add_run(line if line else " "), east_asia=FONT_SONG,
                     latin=FONT_MONO, size=CODE_SIZE)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    set_run_font(p.add_run(strip_inline(text)), east_asia=FONT_HEI, size=CAPTION_SIZE)


def add_figure(doc: Document, image_path: pathlib.Path, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(image_path), width=IMAGE_WIDTH)


def add_table(doc: Document, header: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for cell, text in zip(table.rows[0].cells, header):
        _fill_cell(cell, text, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            _fill_cell(cell, text, bold=False)

    _style_three_line_table(table)


def _fill_cell(cell, text: str, *, bold: bool) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    _add_inline(p, text, size=TABLE_SIZE, force_bold=bold)


def _style_three_line_table(table) -> None:
    """三线表：顶线、底线 1.5pt，表头线 0.75pt，其余（含所有竖线）一律无框。

    注意：不要边框的边必须写成 ``val="nil"``。写成 ``val="single"`` + ``sz="0"``
    Word 仍可能画出一条发丝线，三线表就会多出格线。
    """
    last_row = len(table.rows) - 1
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            for existing in tc_pr.findall(qn("w:tcBorders")):
                tc_pr.remove(existing)

            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                element = OxmlElement(f"w:{edge}")
                if edge == "top" and row_idx == 0:
                    element.set(qn("w:val"), "single")
                    element.set(qn("w:sz"), "12")  # 1.5pt 顶线
                elif edge == "bottom" and row_idx == last_row:
                    element.set(qn("w:val"), "single")
                    element.set(qn("w:sz"), "12")  # 1.5pt 底线
                elif edge == "bottom" and row_idx == 0:
                    element.set(qn("w:val"), "single")
                    element.set(qn("w:sz"), "6")  # 0.75pt 表头线
                else:
                    element.set(qn("w:val"), "nil")
                element.set(qn("w:space"), "0")
                element.set(qn("w:color"), "000000")
                borders.append(element)
            tc_pr.append(borders)


# ---------------------------------------------------------------- 行内解析

INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def _add_inline(paragraph, text: str, *, size=BODY_SIZE, force_bold=False) -> None:
    """处理 **粗体** 与 `等宽`，其余按正文字体输出。

    粗体里可能还嵌着等宽片段，因此对粗体内部递归调用本函数。
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # 链接只保留文字
    for piece in INLINE_RE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            _add_inline(paragraph, piece[2:-2], size=size, force_bold=True)
        elif piece.startswith("`") and piece.endswith("`"):
            set_run_font(paragraph.add_run(piece[1:-1]), east_asia=FONT_SONG,
                         latin=FONT_MONO, size=CODE_SIZE)
        else:
            set_run_font(paragraph.add_run(piece), size=size, bold=force_bold)


# ---------------------------------------------------------------- Markdown 解析

CAPTION_RE = re.compile(r"^\*\*(表\s*\d+.*?)\*\*$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$")
OL_RE = re.compile(r"^\d+\.\s+(.*)$")
UL_RE = re.compile(r"^[-*]\s+(.*)$")
SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def convert(md_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    lines = md_path.read_text(encoding="utf-8").split("\n")
    doc = setup_document()

    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), DOC_TITLE)
    add_cover(doc, title)
    add_toc(doc)

    # 正文另起一节，便于单独加页码
    body_section = doc.add_section(WD_SECTION.NEW_PAGE)
    body_section.top_margin = Cm(2.54)
    body_section.bottom_margin = Cm(2.54)
    body_section.left_margin = Cm(3.17)
    body_section.right_margin = Cm(3.17)
    add_page_number_footer(body_section)
    restart_page_numbering(body_section, start=1)

    table_no = figure_no = 0
    pending_caption = ""
    seen_title = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # ---- 一级标题（# ）只用于封面，正文里跳过 ----
        if stripped.startswith("# ") and not seen_title:
            seen_title = True
            i += 1
            continue

        # ---- 代码块 ----
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code_block(doc, block)
            continue

        # ---- 表格 ----
        if stripped.startswith("|") and i + 1 < len(lines) and SEPARATOR_RE.match(lines[i + 1].strip()):
            header = split_row(stripped)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            table_no += 1
            caption = pending_caption or f"表 {table_no}"
            add_caption(doc, caption)
            pending_caption = ""
            add_table(doc, header, rows)
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
            continue

        # ---- 图片 ----
        match = IMAGE_RE.match(stripped)
        if match:
            alt, rel_path = match.group(1), match.group(2)
            image_path = (md_path.parent / rel_path)
            if not image_path.is_file():
                print(f"[warn] 找不到图片，跳过：{rel_path}")
            else:
                figure_no += 1
                add_figure(doc, image_path, "")
                add_caption(doc, f"图 {figure_no}　{alt}")
            i += 1
            continue

        # ---- 题注（下一个元素若是表格就用它） ----
        match = CAPTION_RE.match(stripped)
        if match:
            pending_caption = match.group(1)
            i += 1
            continue

        # ---- 标题 ----
        match = HEADING_RE.match(stripped)
        if match:
            add_heading(doc, match.group(2).strip(), len(match.group(1)) - 1)
            i += 1
            continue

        # ---- 引用块（空的 > 行分隔成多个段落）----
        if stripped.startswith(">"):
            group: list[list[str]] = [[]]
            while i < len(lines) and lines[i].strip().startswith(">"):
                content = lines[i].strip().lstrip(">").strip()
                if content:
                    group[-1].append(content)
                else:
                    group.append([])
                i += 1
            for block in (b for b in group if b):
                add_quote(doc, block)
            continue

        # ---- 列表（可能跨行，需要把续行并进同一个条目）----
        match = OL_RE.match(stripped) or UL_RE.match(stripped)
        if match:
            ordered = OL_RE.match(stripped) is not None
            block = [match.group(1).strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if (not nxt or nxt.startswith(("#", "|", ">", "```", "-", "*"))
                        or OL_RE.match(nxt) or UL_RE.match(nxt) or IMAGE_RE.match(nxt)):
                    break
                block.append(nxt)
                i += 1
            add_list_item(doc, "".join(block), ordered=ordered)
            continue

        # ---- 分隔线 ----
        if stripped in ("---", "***", "___"):
            i += 1
            continue

        # ---- 空行 ----
        if not stripped:
            i += 1
            continue

        # ---- 普通段落：连续非空行合并 ----
        block = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", ">", "```", "-", "*"))
                    or OL_RE.match(nxt) or IMAGE_RE.match(nxt)):
                break
            block.append(nxt)
            i += 1
        add_body_paragraph(doc, "".join(block))

    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    output = convert(SOURCE_MD, OUTPUT_DOCX)
    print(f"已生成: {output}")
    print(f"学号 {STUDENT['student_id']} | 姓名 {STUDENT['name']} | 班级 {STUDENT['class_name']}")