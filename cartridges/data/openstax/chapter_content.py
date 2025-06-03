from dataclasses import dataclass

from dataclasses import dataclass
import re

@dataclass
class TextbookSection:
    title: str
    content: str

def clean_xml(xml_text):
    """
    Remove id attributes, figure tags and image tags from XML text,
    while preserving content between tags.
    
    Args:
        xml_text (str): The XML content to clean
        
    Returns:
        str: Cleaned XML content
    """
    # Remove id attributes
    cleaned = re.sub(r'\s+id="[^"]*"', '', xml_text)
    cleaned = re.sub(r'\s+target-id="[^"]*"', '', cleaned)
    
    # Remove only the figure opening and closing tags, keeping their content
    cleaned = re.sub(r'<figure[^>]*>', '', cleaned)
    cleaned = re.sub(r'</figure>', '', cleaned)
    
    # Remove image tags (self-closing tags)
    cleaned = re.sub(r'<image[^>]*?/>', '', cleaned)
    
    return cleaned

def extract_title(xml_text):
    """
    Extract the main title from XML content
    
    Args:
        xml_text (str): The XML content
        
    Returns:
        str: The extracted title or None if not found
    """
    title_match = re.search(r'<title>(.*?)</title>', xml_text)
    
    if title_match:
        return title_match.group(1)  # Return the captured content
    else:
        return "Unknown Title"


def parse_xml(content: str) -> TextbookSection:
    return TextbookSection(
        content=clean_xml(content),
        title=extract_title(content),
    )