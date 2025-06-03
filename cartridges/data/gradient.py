

from pathlib import Path


def gradient_wiki_content():

    return (Path(__file__).resolve().parent / 'gradient_wiki.txt').read_text()
