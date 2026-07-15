import re

_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

CHAPTER_PATTERN = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)章")
ARTICLE_PATTERN = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)条")
SECTION_PATTERN = re.compile(r"第([0-9零〇一二两三四五六七八九十百千万]+)节")
DOCUMENT_FILE_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,80}\.(?:pdf|docx?|txt|md|xlsx?))",
    re.IGNORECASE,
)
QUOTED_DOCUMENT_HINT_PATTERN = re.compile(
    r"[\"'“”‘’《》]([\u4e00-\u9fffA-Za-z0-9_\- ]{2,80})[\"'“”‘’《》]"
)
DOCUMENT_CONTEXT_HINT_PATTERN = re.compile(
    r"^(?:概括|总结|查看|查询|关于|对比|比较|阅读)?"
    r"([\u4e00-\u9fffA-Za-z0-9_\- ]{2,80})"
    r"(?:的)?(?:主要内容|不同点|相同点|区别|全文|内容)?$"
)
HEADING_HINT_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,30})(?:章节|章|节|标题|部分|制度|说明|流程)"
)
EXACT_TOKEN_PATTERNS = (
    re.compile(r"\b[A-Z]{1,10}-\d{2,}\b"),
    re.compile(r"\b(?:ERR|ERROR|HTTP|CODE)[-_ ]?\d{2,}\b", re.IGNORECASE),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-_][A-Za-z0-9]+)?\b"),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)"),
)
YEAR_RANGE_PATTERN = re.compile(
    r"(20\d{2}|19\d{2})\s*(?:年)?\s*[至到\-~—]\s*(20\d{2}|19\d{2})"
)
YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})年?")
NEGATIVE_PATTERN = re.compile(
    r"(?:不要|排除|不包括|不含|除了|非)([\u4e00-\u9fffA-Za-z0-9_\-]{2,20})"
)
ENTITY_PATTERNS = {
    "law": re.compile(r"[\u4e00-\u9fff]{2,30}法"),
    "organization": re.compile(
        r"[\u4e00-\u9fffA-Za-z0-9]{2,30}(?:公司|部门|委员会|协会|中心|集团)"
    ),
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
}

STOP_WORDS = {
    "什么",
    "一下",
    "请问",
    "介绍",
    "说明",
    "主要",
    "内容",
    "区别",
    "对比",
    "总结",
    "概括",
    "不要",
    "排除",
}


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).lower()


def parse_chinese_or_arabic_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS.get(char)
        if unit is None:
            return None
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = 0
        else:
            section += (number or 1) * unit
        number = 0
    return total + section + number


def unique_non_empty(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = value.strip(" ，。？！、:：；;（）()[]【】\"'")
        if item and item not in result:
            result.append(item)
    return result
