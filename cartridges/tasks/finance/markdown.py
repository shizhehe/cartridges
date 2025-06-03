from dataclasses import dataclass
from typing import List, Optional
import re
import os

@dataclass
class MarkdownSection:
    level: int 
    title: str
    name: str
    path: str
    content: str
    desc: str = ""

def _to_camel_case(title: str) -> str:
    name = title.replace("_", "__").replace(" ", "_").lower()
    # Remove non alphanumeric characters except underscores
    import string
    name = ''.join(char for char in name if char.isalnum() or char == '_')
    return name

def _build_path(current_section: MarkdownSection, level: int, name: str) -> str:
    if level > current_section.level:
    
    
        return os.path.join(
            current_section.path,
            *["_" for _ in range(level - current_section.level - 1)],
            name
        )
    else:
        return os.path.join(
            *current_section.path.split("/")[:level],
            name
        )

def markdown_to_sections(text: str, root: str="root") -> List[MarkdownSection]:
    base_section = MarkdownSection(level=0, title=root, path=root, name=root,content="")
    sections: List[MarkdownSection] = [base_section]
    active_sections: List[MarkdownSection] = [base_section]
    current_section = base_section

    lines = text.split("\n\n")

    max_level = 0
    for line in lines:
        if header_match := re.match(r'^(#+)\s*(.*)', line.lstrip()):
            level = len(header_match.group(1))
            max_level = max(max_level, level)
    bold_level = max_level + 1
    
    for line in lines:
        is_header = False
        header_text = None
        level = None

        stripped_line = line.lstrip() 
        # Check for markdown headers (e.g., "# Header", "## Subheader", etc.)
        if stripped_line.startswith('#'):
            header_match = re.match(r'^(#+)\s*(.*)', stripped_line)
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                if header_text:
                    is_header = True

        # check for bold headers
        elif stripped_line.startswith('**') and stripped_line.endswith('**') and stripped_line.count('**') == 2:
            header_text = stripped_line[2:-2].strip()
            if header_text:
                level = bold_level
                is_header = True

        if is_header:
            name = _to_camel_case(header_text)
            path = _build_path(current_section, level, name)
            current_section = MarkdownSection(level=level, title=header_text, name=name, path=path, content="")
            
            sections.append(current_section)
            active_sections = [section for section in active_sections if section.level < level]
            active_sections.append(current_section)
        
        # regardless of whether it's a header or not, we add the line to the content of the active sections
        for active_section in active_sections:
            active_section.content += line + "\n\n"
    sections = [section for section in sections if section.content.replace("\n", "").strip() != ""]
    return sections


        