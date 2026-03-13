#!/usr/bin/env python3
"""
合并梁羽生七剑下天山的htm文件为txt格式
"""

import os
import re
from pathlib import Path
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """从HTML中提取文本内容"""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script_style = False
        self.skip_tags = {'script', 'style', 'head', 'title', 'meta'}
        self.tag_stack = []
        
    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag)
        if tag in self.skip_tags:
            self.in_script_style = True
            
    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        if tag in self.skip_tags:
            self.in_script_style = False
            
    def handle_data(self, data):
        if not self.in_script_style:
            self.text_parts.append(data)
            
    def get_text(self):
        return ''.join(self.text_parts)


def clean_text(text):
    """清理提取的文本"""
    # 替换HTML实体
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    
    # 替换<br>为换行
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    
    # 移除其他HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    return text


def extract_chapter_title(text):
    """从文本中提取章节标题"""
    lines = text.split('\n')
    for line in lines[:20]:  # 只在前20行查找
        line = line.strip()
        # 匹配"楔子"、"第一回"、"第X回"等
        if re.match(r'^(楔子|第[一二三四五六七八九十百千万]+回|第\d+回)$', line):
            return line
    return None


def extract_subtitle(text):
    """提取回目标题（红色字体部分）"""
    # 查找红色字体的回目标题
    match = re.search(r'<font\s+color="red">([^<]+)</font>', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def process_htm_file(filepath):
    """处理单个htm文件，返回章节标题和正文"""
    try:
        # 读取文件，尝试多种编码
        content = None
        for encoding in ['gb18030', 'gb2312', 'gbk', 'utf-8']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            with open(filepath, 'r', encoding='gb18030', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"读取文件失败 {filepath}: {e}")
        return None, None
    
    # 从TITLE中提取章节信息
    title_match = re.search(r'<TITLE>[^-]+-->[^-]+-->\s*([^<]+)</TITLE>', content)
    chapter_title = None
    if title_match:
        chapter_title = title_match.group(1).strip()
    
    # 提取红色回目标题
    subtitle = extract_subtitle(content)
    
    # 提取正文内容（在<TD class="tt2">...</TD>之间）
    content_match = re.search(r'<TD\s+class="tt2"[^>]*>(.*?)</TD>', content, re.DOTALL | re.IGNORECASE)
    if not content_match:
        return None, None
    
    body_content = content_match.group(1)
    
    # 提取章节标题（从正文内容中）
    if not chapter_title:
        chapter_title = extract_chapter_title(body_content)
    
    # 清理正文
    # 移除导航链接
    body_content = re.sub(r'<A\s+HREF="[^"]*"[^>]*>[^<]*</A>', '', body_content, flags=re.IGNORECASE)
    # 移除hr标签
    body_content = re.sub(r'<hr[^>]*>', '', body_content, flags=re.IGNORECASE)
    # 移除center标签但保留内容
    body_content = re.sub(r'</?center>', '', body_content, flags=re.IGNORECASE)
    # 移除font标签但保留内容
    body_content = re.sub(r'</?font[^>]*>', '', body_content, flags=re.IGNORECASE)
    # 移除b标签但保留内容
    body_content = re.sub(r'</?b>', '', body_content, flags=re.IGNORECASE)
    # 移除strong标签但保留内容
    body_content = re.sub(r'</?strong>', '', body_content, flags=re.IGNORECASE)
    
    # 提取文本
    text = clean_text(body_content)
    
    # 清理文本
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # 跳过空行和特定内容
        if not line:
            continue
        if '------------------' in line:
            continue
        if '一鸣扫描' in line or '雪儿校对' in line:
            continue
        if '后一页' in line or '前一页' in line or '回目录' in line:
            continue
        if line.startswith('(') and ')' in line and len(line) < 50:
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 合并章节标题和副标题
    final_title = chapter_title
    if subtitle and subtitle != chapter_title:
        if final_title:
            final_title = f"{final_title} {subtitle}"
        else:
            final_title = subtitle
    
    return final_title, text


def main():
    # 设置路径
    base_dir = Path(r'c:\Users\cthul\Desktop\scholar\data\extend\梁羽生\qjxt')
    output_file = Path(r'c:\Users\cthul\Desktop\scholar\data\extend\梁羽生\梁羽生-七剑下天山.txt')
    
    # 获取所有htm文件（排除index.html）
    htm_files = sorted([f for f in base_dir.glob('*.htm') if f.name != 'index.html'])
    
    print(f"找到 {len(htm_files)} 个htm文件")
    
    # 合并内容
    all_content = []
    
    for htm_file in htm_files:
        print(f"处理: {htm_file.name}")
        chapter_title, text = process_htm_file(htm_file)
        
        if text:
            if chapter_title:
                all_content.append(f"{chapter_title}")
                all_content.append("")
            all_content.append(text)
            all_content.append("")
            all_content.append("")
    
    # 添加结束标记
    all_content.append("<|endoftext|>")
    
    # 写入输出文件，使用UTF-8编码
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_content))
    
    print(f"\n合并完成！输出文件: {output_file}")
    print(f"总字符数: {len(''.join(all_content))}")


if __name__ == '__main__':
    main()
