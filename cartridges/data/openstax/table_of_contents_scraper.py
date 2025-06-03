import xml.etree.ElementTree as ET
import json
from dataclasses import dataclass

import requests

from capsules.data.openstax.structs import ChapterMetadata, TextbookMetadata

# Define namespace mapping.
ns = {"col": "http://cnx.rice.edu/collxml", "md": "http://cnx.rice.edu/mdml"}


def extract_leaf_subcollections(root) -> list[ChapterMetadata]:
    """
    Extracts leaf subcollections from the XML root.
    A leaf subcollection is defined as a <col:subcollection> that, in its <col:content>,
    contains modules (i.e. <col:module>) and no nested <col:subcollection> elements.

    Returns a list of dictionaries with keys: number, title, and modules.
    """
    results = []
    count = 0

    def process_subcollection(elem):
        nonlocal count
        # Find the <col:content> element of this subcollection.
        content = elem.find("col:content", ns)
        if content is None:
            return
        # Look for any nested subcollections.
        child_subcollections = content.findall("col:subcollection", ns)
        if child_subcollections:
            # Not a leaf, process each nested subcollection.
            for sub in child_subcollections:
                process_subcollection(sub)
        else:
            # This is a leaf subcollection.
            title_elem = elem.find("md:title", ns)
            title = title_elem.text if title_elem is not None else "No Title"
            # Extract the module document IDs from direct <col:module> children.
            modules = [
                mod.attrib.get("document") for mod in content.findall("col:module", ns)
            ]
            count += 1
            results.append(
                ChapterMetadata(
                    number=count,
                    title=title,
                    modules_ids=modules,
                )
            )

    # Start processing: find the root <col:content> element.
    content_root = root.find("col:content", ns)
    if content_root is not None:
        # Process every subcollection under the root.
        for sub in content_root.findall("col:subcollection", ns):
            process_subcollection(sub)
    return results


def get_chapters(metadata: TextbookMetadata) -> list[ChapterMetadata]:
    collection_url = f"https://raw.githubusercontent.com/openstax/{metadata.github_repo}/refs/heads/main/collections/{metadata.collection_name}"
    root = ET.fromstring(requests.get(collection_url).text)
    extracted_data = extract_leaf_subcollections(root)
    return extracted_data
