import re

import requests

from capsules.data.openstax.structs import (
    ChapterMetadata,
    TextbookMetadata,
    TextbookSection,
)


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
    cleaned = re.sub(r'\s+id="[^"]*"', "", xml_text)
    # TODO: check this
    cleaned = re.sub(r'\s+figure-id="[^"]*"', "", cleaned)

    # Remove only the figure opening and closing tags, keeping their content
    cleaned = re.sub(r"<figure[^>]*>", "", cleaned)
    cleaned = re.sub(r"</figure>", "", cleaned)

    # Remove image tags (self-closing tags)
    cleaned = re.sub(r"<image[^>]*?/>", "", cleaned)

    return cleaned


def extract_title(xml_text):
    """
    Extract the main title from XML content

    Args:
        xml_text (str): The XML content

    Returns:
        str: The extracted title or None if not found
    """
    title_match = re.search(r"<title>(.*?)</title>", xml_text)

    if title_match:
        return title_match.group(1)  # Return the captured content
    else:
        return "Unknown"


def scrape_textbook_chapter(
    metadata: TextbookMetadata, chapter_metadata: ChapterMetadata
) -> list[TextbookSection]:
    res = []
    for module in chapter_metadata.modules_ids:
        content = requests.get(
            f"https://raw.githubusercontent.com/openstax/{metadata.github_repo}/refs/heads/main/modules/{module}/index.cnxml"
        ).text

        res.append(
            TextbookSection(
                title=extract_title(content),
                content=clean_xml(content),
            )
        )

    return res
