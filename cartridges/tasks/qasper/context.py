from typing import List, Optional

from datasets import load_dataset

from capsules.generate.run import BaseContextConfig
from capsules.context import StructuredContext

TOPIC_TO_IDS = {
    "question": [
        '1908.06606',
        '1704.05572',
        '1905.08949',
        '1808.09920',
        '1603.01417',
        '1808.03986',
        '1907.08501',
        '1603.07044',
        '1903.00172',
        '1912.01046',
        '1909.00542',
        '1811.08048',
        '2004.02393',
        '1703.06492',
        '1607.06275',
        '1703.04617'
    ]
}

class QasperStructuredContextConfig(BaseContextConfig):
    topic: str = "question"
    
    def instantiate(self) -> StructuredContext:

        dataset = load_dataset("allenai/qasper", split="train")
        df = dataset.to_pandas()

        paper_ids = TOPIC_TO_IDS[self.topic]
        df = df[df["id"].isin(paper_ids)]
        assert len(df) == len(paper_ids)

        papers = []
        for row in df.to_dict(orient="records"):
            sections = []
            for section_idx, (section_title, paragraphs) in enumerate(zip(
                row["full_text"]["section_name"], row["full_text"]["paragraphs"]
            )):
                sections.append(Section(
                    title=section_title,
                    section_number=section_idx,
                    paragraphs=paragraphs.tolist()
                ))
            paper = Paper(
                id=row["id"],
                title=row["title"],
                abstract=row["abstract"],
                sections=sections
            )
            papers.append(paper)

        return Papers(
            topic=self.topic,
            papers=papers
        )

SECTION_TEMPLATE = """\
<section>
<section-title>{title}</section-title>
<section-number>{section_number}</section-number>
<paragraphs>
{paragraphs}
</paragraphs>
</section>
"""

class Section(StructuredContext):
    title: str
    section_number: int
    paragraphs: List[str]

    @property
    def text(self) -> str:
        paragraph_divider = "\n\n"
        return SECTION_TEMPLATE.format(
            title=self.title,
            section_number=self.section_number,
            paragraphs=paragraph_divider.join(self.paragraphs)
        )


PAPER_TEMPLATE = """\
<title>{title}</title>
<abstract>{abstract}</abstract>
<sections>
{sections}
</sections>
"""

class Paper(StructuredContext):
    id: str
    title: str
    abstract: str
    sections: List[Section]

    @property
    def text(self) -> str:
        section_divider = f"\n---Paper Title: {self.title}---\n"
        return PAPER_TEMPLATE.format(
            title=self.title,
            abstract=self.abstract,
            sections=section_divider.join([section.text for section in self.sections])
        )

PAPERS_TEMPLATE = """\
<topic>{topic}</topic>
<papers>
{papers}
</papers>
"""

class Papers(StructuredContext):
    topic: str
    papers: List[Paper]

    @property
    def text(self) -> str:
        divider = "\n\n-----------\n\n"
        return PAPERS_TEMPLATE.format(
            topic=self.topic,
            papers=divider.join([paper.text for paper in self.papers])
        )
