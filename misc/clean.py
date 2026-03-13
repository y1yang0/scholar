#!/usr/bin/env python3
"""
可扩展的数据清洗脚本

功能模块：
- 修改类：去除连续空行、去除行首空格、添加结束标记等
- 报告类：检测特殊符号等（只报告不修改）

使用方法：
    python clean.py [选项]

示例：
    python clean.py --all           # 执行所有清洗任务
    python clean.py --clean         # 只执行修改类任务
    python clean.py --report        # 只执行报告类任务
    python clean.py --dry-run       # 预览模式，不实际修改文件
"""

import argparse
import os
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================================
# 配置
# ============================================================================

# 数据目录（相对于项目根目录，支持递归处理）
DATA_DIRS = [
    # "data/small
    "data/extend/综合"
]

FILE_PATTERN = "*.txt"
END_OF_TEXT_TOKEN = "<|endoftext|>"

# 特殊符号定义（用于报告）
# 允许的字符集合
ALLOWED_CHARS = set(
    # 基本中文字符
    [chr(c) for c in range(0x4E00, 0x9FFF + 1)]
    +
    # 中文标点
    [chr(c) for c in range(0x3000, 0x303F + 1)]
    +
    # 全角字符
    [chr(c) for c in range(0xFF00, 0xFFEF + 1)]
    +
    # 常用标点和符号（包括各种引号）
    list(
        '，。！？；：""' '「」『』（）【】《》〈〉、—…～·""' "\u201c\u201d\u2018\u2019"
    )
    +
    # 空白字符
    list(" \t\n\r")
    +
    # ASCII字母数字和常用符号
    list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.,:;!?'\"()[]<>|"
    )
)

# 排除的问题字符（虽在全角范围但需要报告）
PROBLEM_CHARS = set("＊")
ALLOWED_CHARS -= PROBLEM_CHARS


# ============================================================================
# 基础类
# ============================================================================


@dataclass
class CleanResult:
    """清洗结果"""

    file_path: str
    task_name: str
    modified: bool = False
    message: str = ""
    details: List[str] = field(default_factory=list)


class CleanTask(ABC):
    """清洗任务基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """任务名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """任务描述"""
        pass

    @property
    @abstractmethod
    def is_modifier(self) -> bool:
        """是否修改文件（True=修改类，False=报告类）"""
        pass

    @abstractmethod
    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        """
        处理文件内容

        Args:
            content: 文件内容
            file_path: 文件路径

        Returns:
            (处理后的内容, 清洗结果)
        """
        pass


# ============================================================================
# 修改类任务
# ============================================================================


class RemoveLeadingSpaces(CleanTask):
    """去除每行行首的空格"""

    @property
    def name(self) -> str:
        return "remove_leading_spaces"

    @property
    def description(self) -> str:
        return "去除每行行首的空格"

    @property
    def is_modifier(self) -> bool:
        return True

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        lines = content.split("\n")
        new_lines = []
        modified_count = 0

        for line in lines:
            # 去除行首空格（包括全角空格）
            stripped = line.lstrip(" \t\u3000")
            if stripped != line:
                modified_count += 1
            new_lines.append(stripped)

        new_content = "\n".join(new_lines)

        if modified_count > 0:
            result.modified = True
            result.message = f"去除了 {modified_count} 行的行首空格"
        else:
            result.message = "无行首空格需要处理"

        return new_content, result


class NormalizeChapterSpacing(CleanTask):
    """规范化章节空行：章节标题前保留一个空行，去除其他所有空行"""

    # 章节标题正则
    CHAPTER_PATTERN = re.compile(
        r"^(第[一二三四五六七八九十百千万零\d]+[章卷回节集部篇])"
    )

    @property
    def name(self) -> str:
        return "normalize_chapter_spacing"

    @property
    def description(self) -> str:
        return "规范章节空行（标题前一空行，去除其他空行）"

    @property
    def is_modifier(self) -> bool:
        return True

    def _is_chapter_title(self, line: str) -> bool:
        """判断是否为章节标题行"""
        return bool(self.CHAPTER_PATTERN.match(line.strip()))

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        lines = content.split("\n")
        n = len(lines)
        new_lines = []
        removed_blank_count = 0
        added_blank_count = 0

        i = 0
        has_content_before = False  # 是否已有非空内容

        while i < n:
            line = lines[i]
            stripped = line.strip()
            is_blank = stripped == ""

            if is_blank:
                # 检查这个空行后面是否紧跟章节标题
                next_idx = i + 1
                while next_idx < n and lines[next_idx].strip() == "":
                    next_idx += 1

                if next_idx < n and self._is_chapter_title(lines[next_idx].strip()):
                    # 空行后面是章节标题，保留一个空行（如果前面有内容）
                    if has_content_before:
                        new_lines.append("")
                    # 跳过所有连续空行
                    skipped = next_idx - i - 1  # 多余的空行数
                    removed_blank_count += skipped
                    i = next_idx
                    continue
                else:
                    # 空行后面不是章节标题，移除这个空行
                    removed_blank_count += 1
                    i += 1
                    continue

            # 非空行
            is_chapter = self._is_chapter_title(stripped)

            if is_chapter and has_content_before:
                # 章节标题，检查前面是否已有空行
                if not new_lines or new_lines[-1].strip() != "":
                    # 前面没有空行，需要添加
                    new_lines.append("")
                    added_blank_count += 1

            new_lines.append(line)
            has_content_before = True
            i += 1

        # 移除末尾空行（文件末尾不应有空行）
        while new_lines and new_lines[-1].strip() == "":
            new_lines.pop()

        new_content = "\n".join(new_lines)

        # 确保以换行符结尾
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        # 比较时也确保原内容以换行符结尾
        content_normalized = content.rstrip("\n") + "\n" if content.strip() else content

        if new_content != content_normalized:
            result.modified = True
            if removed_blank_count > 0 or added_blank_count > 0:
                result.message = f"移除 {removed_blank_count} 空行，章节前添加 {added_blank_count} 空行"
            else:
                result.message = "规范化空行"
        else:
            result.message = "无空行需要处理"

        return new_content, result


class SplitVolumeChapterLine(CleanTask):
    """拆分同一行中的卷+章标题，如"第三卷 xxx 第一章 yyy"拆成两行"""

    # 匹配同一行中包含卷和章的模式
    # 支持: "第X卷 卷名 第X章 章名" 或 "卷X 卷名 第X章 章名"
    VOLUME_CHAPTER_PATTERN = re.compile(
        r"^((?:第[一二三四五六七八九十百千万零\d]+卷|卷[一二三四五六七八九十百千万零\d]+)[^第]*)"  # 卷部分
        r"(第[一二三四五六七八九十百千万零\d]+[章回节].*)$"  # 章部分
    )

    @property
    def name(self) -> str:
        return "split_volume_chapter_line"

    @property
    def description(self) -> str:
        return "拆分卷+章连在一起的标题为两行"

    @property
    def is_modifier(self) -> bool:
        return True

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        lines = content.split("\n")
        new_lines = []
        split_count = 0

        for line in lines:
            match = self.VOLUME_CHAPTER_PATTERN.match(line.strip())
            if match:
                volume_part = match.group(1).strip()
                chapter_part = match.group(2).strip()
                new_lines.append(volume_part)
                new_lines.append("")  # 空行分隔
                new_lines.append(chapter_part)
                split_count += 1
                result.details.append(f'"{line.strip()}" -> "{volume_part}" + "{chapter_part}"')
            else:
                new_lines.append(line)

        new_content = "\n".join(new_lines)

        if split_count > 0:
            result.modified = True
            result.message = f"拆分了 {split_count} 处卷+章标题"
        else:
            result.message = "无卷+章连写标题需要处理"

        return new_content, result


class RemoveAnnotationBrackets(CleanTask):
    """去除补充性括号说明，如（注：）（附：）（注释：）等"""

    # 匹配补充性括号说明的正则
    # 匹配：（注：xxx）（附：xxx）（注释：xxx）（按：xxx）（译注：xxx）等
    ANNOTATION_PATTERN = re.compile(
        r"[（\(]"  # 开括号（中文或英文）
        r"(注|附|注释|按|译注|编注|编者注|作者注|原注|校注)"  # 关键词
        r"[：:]"  # 冒号（中文或英文）
        r"[^）\)]*"  # 内容（非闭括号的任意字符）
        r"[）\)]"  # 闭括号（中文或英文）
    )

    @property
    def name(self) -> str:
        return "remove_annotation_brackets"

    @property
    def description(self) -> str:
        return "去除补充性括号说明（注：）（附：）等"

    @property
    def is_modifier(self) -> bool:
        return True

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        # 查找所有匹配
        matches = list(self.ANNOTATION_PATTERN.finditer(content))

        if matches:
            # 记录被移除的内容
            removed_examples = []
            for m in matches[:5]:  # 最多显示5个例子
                removed_examples.append(f'"{m.group()}"')

            # 执行替换
            new_content = self.ANNOTATION_PATTERN.sub("", content)

            result.modified = True
            result.message = f"移除了 {len(matches)} 处补充性括号说明"
            result.details = removed_examples
        else:
            new_content = content
            result.message = "无补充性括号说明需要处理"

        return new_content, result


class AddEndOfTextToken(CleanTask):
    """在文件末尾添加结束标记"""

    @property
    def name(self) -> str:
        return "add_endoftext_token"

    @property
    def description(self) -> str:
        return f"在文件末尾添加 {END_OF_TEXT_TOKEN} 标记"

    @property
    def is_modifier(self) -> bool:
        return True

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        # 去除末尾空白
        content = content.rstrip()

        # 检查是否已有结束标记
        if content.endswith(END_OF_TEXT_TOKEN):
            result.message = "已存在结束标记，无需添加"
            return content + "\n", result

        # 添加结束标记
        new_content = content + "\n" + END_OF_TEXT_TOKEN + "\n"
        result.modified = True
        result.message = f"已添加 {END_OF_TEXT_TOKEN} 标记"

        return new_content, result


class NormalizeChapterTitles(CleanTask):
    """标准化章节标题格式"""

    # 阿拉伯数字到中文数字的映射
    ARABIC_TO_CHINESE = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }

    @property
    def name(self) -> str:
        return "normalize_chapter_titles"

    @property
    def description(self) -> str:
        return "标准化章节标题（数字转中文、去除装饰符号、补全章名）"

    @property
    def is_modifier(self) -> bool:
        return True

    def _arabic_to_chinese(self, num_str: str) -> str:
        """将阿拉伯数字转换为中文数字"""
        num = int(num_str)
        if num == 0:
            return "零"

        units = ["", "十", "百", "千", "万"]
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

        if num < 10:
            return digits[num]
        elif num < 100:
            tens, ones = divmod(num, 10)
            if tens == 1:
                return "十" + (digits[ones] if ones else "")
            else:
                return digits[tens] + "十" + (digits[ones] if ones else "")
        elif num < 1000:
            hundreds, remainder = divmod(num, 100)
            tens, ones = divmod(remainder, 10)
            result = digits[hundreds] + "百"
            if tens == 0 and ones == 0:
                return result
            elif tens == 0:
                return result + "零" + digits[ones]
            else:
                return (
                    result
                    + (digits[tens] if tens != 1 else "")
                    + "十"
                    + (digits[ones] if ones else "")
                )
        else:
            # 简化处理：对于更大的数字直接逐位转换
            return "".join(self.ARABIC_TO_CHINESE.get(c, c) for c in num_str)

    def _normalize_title(self, line: str) -> Tuple[str, bool]:
        """
        标准化单行标题
        返回: (标准化后的行, 是否修改)
        """
        original = line

        # 章节关键字
        chapter_keywords = ["章", "卷", "回", "节", "集", "部", "篇"]
        keyword_pattern = "|".join(chapter_keywords)

        # 模式1: 【第X章】、（第X章）、---第X章--- 等带装饰的标题
        decorated_pattern = rf"^[\s　]*[【\[（\(「『\-—=★☆●○\*]+\s*(第[一二三四五六七八九十百千万零\d]+(?:{keyword_pattern}))\s*[】\]）\)」』\-—=★☆●○\*]*\s*(.*)$"
        match = re.match(decorated_pattern, line)
        if match:
            chapter_part = match.group(1)
            title_part = match.group(2).strip()
            # 处理章节部分中的阿拉伯数字
            chapter_part = self._convert_chapter_number(chapter_part)
            if title_part:
                line = f"{chapter_part} {title_part}"
            else:
                line = f"{chapter_part} 无题"

        # 模式2: 第1章、第12章 等阿拉伯数字格式
        elif re.match(rf"^[\s　]*(第)(\d+)({keyword_pattern})\s*(.*)$", line):
            match = re.match(rf"^[\s　]*(第)(\d+)({keyword_pattern})\s*(.*)$", line)
            prefix = match.group(1)
            num = match.group(2)
            keyword = match.group(3)
            title_part = match.group(4).strip()
            chinese_num = self._arabic_to_chinese(num)
            if title_part:
                line = f"{prefix}{chinese_num}{keyword} {title_part}"
            else:
                line = f"{prefix}{chinese_num}{keyword} 无题"

        # 模式3: 标准格式但没有章名 "第一章" -> "第一章 无题"
        elif re.match(rf"^[\s　]*(第[一二三四五六七八九十百千万零]+(?:{keyword_pattern}))[\s　]*$", line):
            match = re.match(rf"^[\s　]*(第[一二三四五六七八九十百千万零]+(?:{keyword_pattern}))[\s　]*$", line)
            chapter_part = match.group(1)
            line = f"{chapter_part} 无题"

        # 模式4: 纯数字章节 "1. xxx" 或 "1、xxx"
        elif re.match(r"^[\s　]*(\d+)[\.、\s]\s*(.*)$", line):
            match = re.match(r"^[\s　]*(\d+)[\.、\s]\s*(.*)$", line)
            num = match.group(1)
            title_part = match.group(2).strip()
            # 只处理看起来像章节号的（1-999）
            if 1 <= int(num) <= 999:
                chinese_num = self._arabic_to_chinese(num)
                if title_part:
                    line = f"第{chinese_num}章 {title_part}"
                else:
                    line = f"第{chinese_num}章 无题"

        # 后处理: 处理标题中的括号数字 （1）（2）-> （一）（二）
        def replace_bracket_num(m):
            num = int(m.group(1))
            chinese = self._arabic_to_chinese(str(num))
            # 保持原括号类型
            if m.group(0).startswith("（"):
                return f"（{chinese}）"
            else:
                return f"({chinese})"

        # 匹配中文或英文括号内的数字
        bracket_num_pattern = r"[（\(](\d+)[）\)]"
        line = re.sub(bracket_num_pattern, replace_bracket_num, line)

        return line, line != original

    def _convert_chapter_number(self, chapter_str: str) -> str:
        """转换章节字符串中的阿拉伯数字为中文"""

        def replace_num(m):
            return self._arabic_to_chinese(m.group(0))

        return re.sub(r"\d+", replace_num, chapter_str)

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        lines = content.split("\n")
        new_lines = []
        modified_count = 0
        modified_details = []

        for line in lines:
            new_line, modified = self._normalize_title(line)
            if modified:
                modified_count += 1
                modified_details.append(f'"{line.strip()}" -> "{new_line}"')
            new_lines.append(new_line)

        new_content = "\n".join(new_lines)

        if modified_count > 0:
            result.modified = True
            result.message = f"标准化了 {modified_count} 个章节标题"
            result.details = modified_details  # 输出所有变化
        else:
            result.message = "无需标准化的章节标题"

        return new_content, result


# ============================================================================
# 报告类任务
# ============================================================================


class ReportSpecialChars(CleanTask):
    """报告特殊符号（不修改文件）"""

    @property
    def name(self) -> str:
        return "report_special_chars"

    @property
    def description(self) -> str:
        return "检测并报告特殊符号所在位置"

    @property
    def is_modifier(self) -> bool:
        return False

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        # 查找特殊符号（不在允许字符集中的字符）
        special_chars = {}
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for col, char in enumerate(line, 1):
                if char not in ALLOWED_CHARS:
                    if char not in special_chars:
                        special_chars[char] = []
                    # 只记录前3个位置
                    if len(special_chars[char]) < 3:
                        start = max(0, col - 11)
                        end = min(len(line), col + 10)
                        context = line[start:end]
                        special_chars[char].append(
                            f"行{line_num}列{col}: ...{context}..."
                        )

        if special_chars:
            result.message = f"发现 {len(special_chars)} 种特殊符号"
            for char, locations in special_chars.items():
                char_repr = repr(char)
                result.details.append(f"  符号 {char_repr} (U+{ord(char):04X}):")
                for loc in locations:
                    result.details.append(f"    {loc}")
                total = content.count(char)
                if total > len(locations):
                    result.details.append(f"    ... 共 {total} 处")
        else:
            result.message = "未发现特殊符号"

        return content, result  # 报告类不修改内容


class ReportProblemKeywords(CleanTask):
    """报告高频问题词（不修改文件）"""

    # 章节标题正则（用于排除标题行的文末标记检测）
    CHAPTER_PATTERN = re.compile(
        r"^(第[一二三四五六七八九十百千万零\d]+[章卷回节集部篇]|终章|大结局)"
    )

    # 即使在章节标题行中也要检测的模式（如"第五十二章本卷终"）
    ALWAYS_DETECT_PATTERNS = [
        r"本书[完终]",
        r"本卷[完终]",
        r"本章[完终]",
        r"本集[完终]",
        r"本部[完终]",
        r"本篇[完终]",
    ]

    # 高频问题词分类定义
    PROBLEM_PATTERNS = {
        "文末标记": [
            r"全文完",
            r"全书完",
            r"正文完",
            r"本书完",
            r"本卷完",
            r"本章完",
            r"本集完",
            r"本部完",
            r"本篇完",
            r"全书终",
            r"全文终",
            r"全剧终",
            r"^完结$",
            r"终章",
            r"大结局",
            r"第?[一二三四五六七八九十百千万零\d]+[卷章部集篇]终",
            r"[卷章部集篇][一二三四五六七八九十百千万零\d]+终",
            r"（完）",
            r"\(完\)",
            r"——完——",
            r"==完==",
            r"〔完〕",
            r"\[完\]",
            r"【完】",
            r"～完～",
            r"—完—",
        ],
        "网站水印": [
            r"潇湘书院",
            r"起点中文",
            r"纵横中文",
            r"17k",
            r"晋江文学",
            r"创世中文",
            r"读书论坛",
            r"小说网",
            r"文学网",
            r"书客居",
            r"笔趣阁",
            r"顶点小说",
            r"飘天文学",
        ],
        "录入标记": [
            r"手打版",
            r"手打小说",
            r"手工录入",
            r"精校版",
            r"校订版",
            r"电子版",
            r"扫描版",
            r"OCR",
            r"ＯＣＲ",
        ],
        "版权声明": [
            r"版权所有",
            r"转载请",
            r"禁止转载",
            r"授权发布",
            r"盗版必究",
            r"侵权必究",
        ],
        "广告引流": [
            r"更多精彩",
            r"敬请期待",
            r"加入书签",
            r"点击收藏",
            r"推荐票",
            r"月票",
            r"求.*票",
            r"收藏本站",
            r"记住本站",
        ],
        "下载类": [
            r"下载地址",
            r"TXT下载",
            r"txt下载",
            r"免费下载",
            r"电子书下载",
            r"百度云",
            r"网盘",
        ],
        "网址链接": [
            r"www\.",
            r"http://",
            r"https://",
            r"\.com(?![一-龥])",
            r"\.cn(?![一-龥])",
            r"\.net(?![一-龥])",
            r"\.org(?![一-龥])",
        ],
        "格式水印": [
            r"〖[^〗]*阅读[^〗]*〗",
            r"〖[^〗]*章节[^〗]*〗",
            r"≈[^≈]*连载[^≈]*≈",
            r"㈱[^㈱]*㈱",
            r"【[^】]*小说[^】]*】",
            r"【[^】]*阅读[^】]*】",
        ],
    }

    @property
    def name(self) -> str:
        return "report_problem_keywords"

    @property
    def description(self) -> str:
        return "检测高频问题词（网站水印、广告、录入标记等）"

    @property
    def is_modifier(self) -> bool:
        return False

    def _should_skip_line(self, category: str, line: str) -> bool:
        """判断是否跳过该行的检测"""
        if category != "文末标记":
            return False
        if not self.CHAPTER_PATTERN.match(line.strip()):
            return False
        # 章节标题行，但检查是否包含“总是检测”的模式
        for pattern in self.ALWAYS_DETECT_PATTERNS:
            if re.search(pattern, line):
                return False  # 包含高优先级模式，不跳过
        return True  # 普通章节标题，跳过

    def process(self, content: str, file_path: str) -> Tuple[str, CleanResult]:
        result = CleanResult(file_path=file_path, task_name=self.name)

        lines = content.split("\n")
        found_problems = {}  # {类别: [(行号, 匹配文本, 上下文), ...]}

        for category, patterns in self.PROBLEM_PATTERNS.items():
            for pattern in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                for line_num, line in enumerate(lines, 1):
                    # 文末标记类别：根据规则跳过章节标题行
                    if self._should_skip_line(category, line):
                        continue
                    for match in regex.finditer(line):
                        if category not in found_problems:
                            found_problems[category] = []
                        # 只记录前5个匹配
                        if len(found_problems[category]) < 5:
                            start = max(0, match.start() - 10)
                            end = min(len(line), match.end() + 10)
                            context = line[start:end]
                            found_problems[category].append(
                                (line_num, match.group(), context)
                            )

        if found_problems:
            total_count = sum(len(v) for v in found_problems.values())
            result.message = (
                f"发现 {len(found_problems)} 类问题词，共 {total_count}+ 处"
            )
            for category, matches in found_problems.items():
                result.details.append(f"  [{category}]:")
                for line_num, matched, context in matches:
                    result.details.append(
                        f'    行{line_num}: "{matched}" → ...{context}...'
                    )
                # 统计实际总数
                total = 0
                for pattern in self.PROBLEM_PATTERNS[category]:
                    total += len(re.findall(pattern, content, re.IGNORECASE))
                if total > len(matches):
                    result.details.append(f"    ... 共约 {total} 处")
        else:
            result.message = "未发现高频问题词"

        return content, result  # 报告类不修改内容


# ============================================================================
# 任务注册表（方便扩展）
# ============================================================================

# 所有可用的清洗任务
ALL_TASKS: List[CleanTask] = [
    # 修改类
    RemoveLeadingSpaces(),
    SplitVolumeChapterLine(),  # 先拆分卷+章
    NormalizeChapterSpacing(),
    NormalizeChapterTitles(),
    RemoveAnnotationBrackets(),
    AddEndOfTextToken(),
    # 报告类
    ReportSpecialChars(),
    ReportProblemKeywords(),
]


def get_modifier_tasks() -> List[CleanTask]:
    """获取所有修改类任务"""
    return [t for t in ALL_TASKS if t.is_modifier]


def get_reporter_tasks() -> List[CleanTask]:
    """获取所有报告类任务"""
    return [t for t in ALL_TASKS if not t.is_modifier]


# ============================================================================
# 主逻辑
# ============================================================================


class DataCleaner:
    """数据清洗器"""

    def __init__(self, base_dir: str, dry_run: bool = False, verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        self.results: List[CleanResult] = []
        self.log_file = None

        # dry-run 模式下同时输出到日志文件
        if dry_run:
            log_path = self.base_dir / "clean.log"
            self.log_file = open(log_path, "w", encoding="utf-8")

    def _print(self, msg: str = ""):
        """输出到控制台，dry-run模式下同时写入日志文件"""
        print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def get_all_files(self) -> List[Path]:
        """获取所有待处理文件（递归）"""
        files = []
        for data_dir in DATA_DIRS:
            dir_path = self.base_dir / data_dir
            if dir_path.exists():
                # 使用 ** 递归匹配所有子目录
                files.extend(dir_path.rglob(FILE_PATTERN))
        return sorted(files)

    def process_file(
        self, file_path: Path, tasks: List[CleanTask]
    ) -> List[CleanResult]:
        """处理单个文件"""
        results = []

        # 读取文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            result = CleanResult(
                file_path=str(file_path),
                task_name="read_file",
                message=f"读取失败: {e}",
            )
            return [result]

        original_content = content
        file_modified = False

        # 执行所有任务
        for task in tasks:
            content, result = task.process(content, str(file_path))
            results.append(result)
            if result.modified:
                file_modified = True

        # 写回文件（仅当有修改且非预览模式）
        if file_modified and not self.dry_run:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                result = CleanResult(
                    file_path=str(file_path),
                    task_name="write_file",
                    message=f"写入失败: {e}",
                )
                results.append(result)

        return results

    def run(self, tasks: List[CleanTask]) -> None:
        """执行清洗任务"""
        files = self.get_all_files()

        if not files:
            self._print("未找到待处理文件")
            return

        self._print(f"{'='*60}")
        self._print(f"数据清洗任务")
        self._print(f"{'='*60}")
        self._print(f"模式: {'预览模式（不修改文件）' if self.dry_run else '执行模式'}")
        self._print(f"文件数: {len(files)}")
        self._print(f"任务列表:")
        for task in tasks:
            task_type = "修改" if task.is_modifier else "报告"
            self._print(f"  - [{task_type}] {task.description}")
        self._print(f"{'='*60}\n")

        # 分离修改类和报告类任务
        modifier_tasks = [t for t in tasks if t.is_modifier]
        reporter_tasks = [t for t in tasks if not t.is_modifier]

        modified_count = 0
        report_files = []

        for file_path in files:
            rel_path = file_path.relative_to(self.base_dir)

            # 执行修改类任务
            if modifier_tasks:
                results = self.process_file(file_path, modifier_tasks)
                self.results.extend(results)

                file_modified = any(r.modified for r in results)
                if file_modified:
                    modified_count += 1
                    if self.verbose:
                        self._print(f"[修改] {rel_path}")
                        for r in results:
                            if r.modified:
                                self._print(f"       {r.message}")
                                # 输出详细变化（如章节标题的前后对比）
                                for detail in r.details:
                                    self._print(f"         {detail}")

            # 执行报告类任务
            if reporter_tasks:
                results = self.process_file(file_path, reporter_tasks)
                self.results.extend(results)

                has_report = any(r.details for r in results)
                if has_report:
                    report_files.append((rel_path, results))

        # 打印报告汇总
        if reporter_tasks and report_files:
            self._print(f"\n{'='*60}")
            self._print("特殊符号报告")
            self._print(f"{'='*60}")
            for rel_path, results in report_files:
                self._print(f"\n[文件] {rel_path}")
                for r in results:
                    if r.details:
                        self._print(f"  {r.message}")
                        for detail in r.details:
                            self._print(f"  {detail}")

        # 打印总结
        self._print(f"\n{'='*60}")
        self._print("执行总结")
        self._print(f"{'='*60}")
        self._print(f"处理文件: {len(files)} 个")
        if modifier_tasks:
            action = "将修改" if self.dry_run else "已修改"
            self._print(f"{action}文件: {modified_count} 个")
        if reporter_tasks:
            self._print(f"有问题的文件: {len(report_files)} 个")

        if self.dry_run and modified_count > 0:
            self._print(
                f"\n提示: 使用 --no-dry-run 或移除 --dry-run 参数以实际执行修改"
            )

        # 日志文件提示
        if self.log_file:
            log_path = self.base_dir / "clean.log"
            self._print(f"\n日志已保存至: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="数据清洗脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python clean.py --all             执行所有任务
  python clean.py --clean           只执行修改类任务
  python clean.py --report          只执行报告类任务  
  python clean.py --dry-run --all   预览所有任务（不实际修改）
  python clean.py --list            列出所有可用任务
        """,
    )

    parser.add_argument("--all", action="store_true", help="执行所有任务")
    parser.add_argument("--clean", action="store_true", help="只执行修改类任务")
    parser.add_argument("--report", action="store_true", help="只执行报告类任务")
    parser.add_argument(
        "--dry-run", action="store_true", help="预览模式，不实际修改文件"
    )
    parser.add_argument("--list", action="store_true", help="列出所有可用任务")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式，减少输出")

    args = parser.parse_args()

    # 列出任务
    if args.list:
        print("可用的清洗任务:")
        print("\n修改类任务:")
        for task in get_modifier_tasks():
            print(f"  - {task.name}: {task.description}")
        print("\n报告类任务:")
        for task in get_reporter_tasks():
            print(f"  - {task.name}: {task.description}")
        return

    # 确定要执行的任务
    tasks = []
    if args.all:
        tasks = ALL_TASKS.copy()
    elif args.clean:
        tasks = get_modifier_tasks()
    elif args.report:
        tasks = get_reporter_tasks()
    else:
        parser.print_help()
        return

    # 获取项目根目录（misc的父目录）
    base_dir = Path(__file__).parent.parent

    # 执行清洗
    cleaner = DataCleaner(
        base_dir=str(base_dir), dry_run=args.dry_run, verbose=not args.quiet
    )
    try:
        cleaner.run(tasks)
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()
