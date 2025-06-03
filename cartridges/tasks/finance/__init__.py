from dataclasses import dataclass
import random
from typing import Any, Dict, List, Literal, Optional

from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import pandas as pd

from cartridges.context import StructuredContext
from cartridges.structs import (
    SECTION_BUFFER_TOKENS,
    Message,
    Section,
    SectionedContext,
    Context,
    ContextConvo,
)
from cartridges.datasets import (
    CartridgeDataset,
    CartridgeGenerateDataset,
    CartridgeGenerateDatasetElement,
    TEMPLATE,
)

from cartridges.context import BaseContextConfig
from cartridges.generate.generate_training import BaseSectionedContextConfig
from cartridges.tasks.finance.dataset import load_finance
from cartridges.tasks.finance.markdown import markdown_to_sections, MarkdownSection
from cartridges.tasks.mtob.context import SimpleStructuredContext
from cartridges.transforms import ConvoTransformConfig
from cartridges.utils import get_logger


class MarkdownSectionStructured(StructuredContext):
    title: str
    name: str
    content: str


class FinanceBenchDocumentStructured(StructuredContext):
    sections: List[MarkdownSection]


class FinanceBenchDocumentStructuredConfig(BaseContextConfig):
    doc_names: Optional[List[str]] = None

    def instantiate(self) -> FinanceBenchDocumentStructured:
        assert len(self.doc_names) == 1, "Only one document name is supported"
        df = load_finance(doc_names=self.doc_names)

        for _, row in df.drop_duplicates("doc_name").iterrows():
            return SimpleStructuredContext(content=row["md_text"])

            sections = markdown_to_sections(row["md_text"], root=row["doc_name"])
            documents = []
            for section in sections:
                documents.extend(
                    [
                        MarkdownSectionStructured(
                            title=section.title,
                            content=section.content,
                            name=section.name,
                        )
                    ]
                )

            # breakpoint()
            # return FinanceBenchDocumentStructured(sections=documents)
        assert False

    
class FinanceBenchDocumentStructuredConfigCompose(BaseContextConfig):
    doc_names: Optional[List[str]] = None

    def instantiate(self) -> FinanceBenchDocumentStructured:
        df = load_finance(doc_names=self.doc_names)

        all_content = []
        for _, row in df.drop_duplicates("doc_name").iterrows():
            all_content.append(row["md_text"])
        return SimpleStructuredContext(content="\n".join(all_content))
    
        # all_content = []
        # for _, row in df.drop_duplicates("doc_name").iterrows():
        #     all_content.append(SimpleStructuredContext(content=row["md_text"]))
        # return all_content



class FinanceBenchContextConfig(BaseContextConfig):

    doc_names: Optional[List[str]] = None
    split_on: Literal["markdown", "page", "none"] = "markdown"

    def instantiate(self) -> Context:

        df = load_finance(doc_names=self.doc_names)

        documents: List[Section] = []
        for _, row in df.drop_duplicates("doc_name").iterrows():
            if self.split_on == "markdown":
                sections = markdown_to_sections(row["md_text"], root=row["doc_name"])
                documents = []
                for section in sections:
                    try:
                        documents.extend(
                            [
                                Section(
                                    title=section.title,
                                    path=section.path,
                                    text=section.content,
                                    content=section.content,
                                    desc=section.desc,
                                )
                            ]
                        )
                    except:
                        breakpoint()
            elif self.split_on == "page":
                pages = row["md_text"].split("-----")
                for page_idx, page in enumerate(pages):
                    documents.append(
                        Section(
                            title=f"Page {page_idx}",
                            content=page,
                            path=f"{row['doc_name']}/{page_idx}",
                            type_="page",
                            text=page,
                        )
                    )
            else:
                documents.append(
                    Section(
                        title=row["doc_name"],
                        content=row["md_text"],
                        path=row["doc_name"],
                        text=row["md_text"],
                    )
                )

        if self.doc_names is not None:
            title = f"FinanceBench-{'-'.join(self.doc_names)}"
        else:
            title = "FinanceBench"

        context = Context(
            title=title,
            documents=documents,
            text=row["md_text"],
            sections=documents,
        )

        return context


class FinanceBenchSectionedContextConfig(BaseSectionedContextConfig):

    doc_name: str

    def instantiate(self, tokenizer) -> SectionedContext:

        df = load_finance(doc_names=[self.doc_name])

        sections = []
        current_pages: list[tuple[int, str]] = []
        current_tokens = 0

        # 1: to exclude BOS token
        count_tokens = lambda str_: len(tokenizer.encode(str_)) - 1

        def make_section():
            assert len(current_pages)

            content = "\n".join([page_content for (_, page_content) in current_pages])

            section = Section(
                content=content,
                # tokens=0,
                desc=f"Pages {current_pages[0][0] + 1} through {current_pages[-1][0] + 1}",
            )
            section.tokens = count_tokens(str(section))
            if not section.tokens <= self.max_tokens_per_section:
                breakpoint()
            assert section.tokens <= self.max_tokens_per_section
            sections.append(section)
            current_pages.clear()
            nonlocal current_tokens
            current_tokens = 0

        for _, row in df.drop_duplicates("doc_name").iterrows():

            pages = row["md_text"].split("-----")
            for page_idx, page in enumerate(pages):
                page_content = f"""<page number={page_idx + 1}>
{page}
</page>
"""
                page_tokens = count_tokens(page_content)
                assert page_tokens < self.max_tokens_per_section

                if (
                    current_tokens + page_tokens + SECTION_BUFFER_TOKENS
                    > self.max_tokens_per_section
                ):
                    make_section()

                current_pages.append((page_idx, page_content))
                current_tokens += page_tokens

            make_section()
            break

        title = self.doc_name

        context = SectionedContext(title=title, sections=sections)

        return context


@dataclass
class FinanceBenchQuestion:
    financebench_id: str

    question: str
    answer: str
    justification: Optional[str]

    doc_name: str
    doc_link: str
    md_text: str

    company: str
    doc_period: int
    doc_type: Literal["10K", "10q", "8K", "Earnings"]
    question_type: Literal["metrics-generated", "domain-relevant", "novel-generated"]
    question_reasoning: str
    evidence: Dict[str, str]
    dataset_subset_label: str
    gics_sector: Literal[
        "Industrials",
        "Communication Services",
        "Information Technology",
        "Utilities",
        "Consumer Discretionary",
        "Materials",
        "Financials",
        "Consumer Staples",
        "Health Care",
    ]
    domain_question_num: str


COT_PROMPT_TEMPLATE = """\
Please answer the question below:
<question>
{question}
</question>

First think step by step, but then provide your final answer within <answer> tags like this:
<answer>
{{your answer}}
</answer>"""

PROMPT_TEMPLATE = "{question}"


class FinanceBenchEvalDataset(CartridgeDataset):
    class Config(CartridgeDataset.Config):
        _pass_as_config = True
        doc_names: Optional[List[str]] = None

        max_questions: Optional[int] = None
        cot: bool = True

        # ignored
        label_type: str = "tokens"
        data_sources: List[str] = []
        convo_transforms: list[ConvoTransformConfig] | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        df = load_finance(doc_names=self.config.doc_names)
        self.context = df["md_text"].iloc[0]

        questions: List[FinanceBenchQuestion] = [
            FinanceBenchQuestion(**row) for _, row in df.iterrows()
        ]

        random.Random(42).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]

        if self.config.cot:
            template = COT_PROMPT_TEMPLATE

            def get_answer(question: FinanceBenchQuestion) -> str:
                if question.justification is None:
                    return f"<answer>{question.answer}</answer>"
                else:
                    return (
                        f"{question.justification}\n<answer>{question.answer}</answer>"
                    )

        else:
            template = PROMPT_TEMPLATE

            def get_answer(question: FinanceBenchQuestion) -> str:
                return question.answer

        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=template.format(question=question.question),
                    ),
                    Message(role="assistant", content=get_answer(question)),
                ],
                type="FinanceEval",
                metadata={
                    "question_id": question.financebench_id,
                },
            )
            for question in questions
        ]

        self.tokenizer = tokenizer

        if self.config.convo_transforms is not None:
            self.convo_transforms = [
                transform.instantiate(context=self.context)
                for transform in self.config.convo_transforms
            ]
        else:
            self.convo_transforms = []


class FinanceBenchGenerateDataset(CartridgeGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        doc_names: Optional[List[str]] = None
        max_questions: Optional[int] = None
        # system_prompt: str | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        df = load_finance(doc_names=self.config.doc_names)

        questions: List[FinanceBenchQuestion] = [
            FinanceBenchQuestion(**row) for _, row in df.iterrows()
        ]

        random.Random(42).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]

        self.questions = questions
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> CartridgeGenerateDatasetElement:
        question: FinanceBenchQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template(
            (
                []
                # if self.config.system_prompt is not None
                # else [{"role": "system", "content": self.config.system_prompt}]
            )
            + [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=question.question),
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.answer,
            convo_id=question.financebench_id,
            metadata={"idx": index},
        )

    def __len__(self):
        return len(self.questions)


class FinanceBenchMemorizationDataset(CartridgeDataset):
    class Config(CartridgeDataset.Config):
        _pass_as_config = True
        doc_names: Optional[List[str]] = None

        max_questions: Optional[int] = None
        cot: bool = True

        # ignored
        label_type: str = "tokens"
        data_sources: List[str] = []
        convo_transforms: list[ConvoTransformConfig] | None = None
        context: str = ""

    def generate_questions(self, df: pd.DataFrame) -> List[FinanceBenchQuestion]:

        # extract random snippets from the document and generate questions.
        # our goal is to test how well the model can regurgitate the text.

        text = df["md_text"].iloc[0].split()
        max_words_per_q = 2_000

        # sample self.config.max_questions random snippets
        # question should be a snippet of text and answer should be the subsequent text
        questions = []
        for i in range(self.config.max_questions):
            start_q = random.randint(0, len(text) // 3)
            end_q = random.randint(start_q + 1, 2 * len(text) // 3)
            if end_q - start_q > max_words_per_q:
                end_q = start_q + max_words_per_q
            start_a = end_q
            end_a = random.randint(start_a + 1, len(text) - 1)
            if end_a - start_a > max_words_per_q:
                end_a = start_a + max_words_per_q

            question = " ".join(text[start_q:end_q])
            answer = " ".join(text[start_a:end_a])

            questions.append(
                FinanceBenchQuestion(
                    financebench_id=f"question-{i}",
                    question=question,
                    answer=answer,
                    justification=None,
                    doc_name=df["doc_name"].iloc[0],
                    doc_link=df["doc_link"].iloc[0],
                    md_text=df["md_text"].iloc[0],
                    company="AMD",
                    doc_period=2022,
                    doc_type="10K",
                    question_type="metrics-generated",
                    question_reasoning="",
                    evidence={},
                    dataset_subset_label="",
                    gics_sector="Information Technology",
                    domain_question_num="",
                )
            )
        return questions

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        df = load_finance(doc_names=self.config.doc_names)

        questions = self.generate_questions(df)
        self.context = df["md_text"].iloc[0]

        random.Random(42).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]

        template = PROMPT_TEMPLATE

        def get_answer(question: FinanceBenchQuestion) -> str:
            return question.answer

        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=template.format(question=question.question),
                    ),
                    Message(role="assistant", content=get_answer(question)),
                ],
                type="FinanceEval",
                metadata={
                    "question_id": question.financebench_id,
                },
            )
            for question in questions
        ]

        self.tokenizer = tokenizer

        if self.config.convo_transforms is not None:
            self.convo_transforms = [
                transform.instantiate(context=self.context)
                for transform in self.config.convo_transforms
            ]
        else:
            self.convo_transforms = []


L3b_qa_pairs = {
    "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22? If the quick ratio is not relevant to measure liquidity, please state that and explain why.": "Based on AMD's FY22 financial data, the company appears to have a reasonably healthy liquidity profile. The quick ratio, a widely used liquidity metric, is 1.57, which is calculated by dividing the sum of cash and cash equivalents, short-term investments, accounts receivable, net, and receivables from related parties by current liabilities.\n\nThe quick ratio is a relevant measure of liquidity, as it provides insight into a company's ability to meet its short-term obligations using its liquid assets. In this case, AMD's quick ratio suggests that the company has sufficient liquid assets to cover its current liabilities, indicating a relatively healthy liquidity profile.\n\nIt's worth noting that the quick ratio is not the only liquidity metric, and other measures such as the current ratio or cash flow metrics may also be relevant. However, based on the provided data, the quick ratio appears to be a useful indicator of AMD's liquidity position.",
    "What are the major products and services that AMD sells as of FY22?": "As of FY22, AMD offers a diverse range of products and services across various markets. The company's primary offerings include server-grade microprocessors and graphics processing units (GPUs) designed for data centers, as well as data processing units (DPUs), Field Programmable Gate Arrays (FPGAs), and Adaptive System-on-Chip (SoC) products.\n\nIn addition to these, AMD also provides CPUs, accelerated processing units (APUs) that combine CPUs and GPUs, and chipsets for desktop and notebook personal computers. The company's discrete GPU offerings cater to various applications, while semi-custom SoC products and development services are also available.\n\nFurthermore, AMD's embedded product portfolio includes CPUs, GPUs, APUs, FPGAs, and Adaptive SoC products, which are designed for a range of applications beyond data centers and personal computers. The company may also sell or license portions of its intellectual property (IP) portfolio on an occasional basis.",
    "What drove revenue change as of the FY22 for AMD?": "The significant revenue growth experienced by AMD in FY22 can be attributed to several key factors. Firstly, the company witnessed a substantial increase in sales of its EPYC server processors, which contributed substantially to the overall revenue growth. Additionally, higher sales of semi-custom products in the Gaming segment also played a crucial role in driving revenue up. Furthermore, the inclusion of Xilinx embedded product sales in the Embedded segment further boosted revenue, as Xilinx's acquisition by AMD expanded the company's technology and product portfolio to include adaptable hardware platforms that enable hardware acceleration and rapid innovation across various technologies.",
    "What drove operating margin change as of the FY22 for AMD? If operating margin is not a useful metric for a company like this, then please state that and explain why.": "The primary factor contributing to the decline in AMD's operating income for the fiscal year 2022 was the amortization of intangible assets resulting from the acquisition of Xilinx, Inc. This significant decrease in operating income was a direct consequence of the acquisition, which expanded AMD's technology and product portfolio, but also led to increased expenses associated with the integration of Xilinx's assets.\n\nIn the context of AMD's diversified business model, operating income may not be the most useful metric for evaluating the company's performance. This is because operating income does not account for the impact of non-operating items, such as interest expenses and foreign exchange fluctuations, which can significantly affect a company's bottom line. Additionally, operating income may not accurately reflect a company's ability to generate cash, as it does not consider the timing of cash inflows and outflows.\n\nIn AMD's case, the company's significant increase in net revenue in 2022, driven by growth in the Data Center, Gaming, and Embedded segments, suggests that the company's diversified business model is generating substantial cash flows. However, the decline in operating income due to amortization of intangible assets associated with the Xilinx acquisition highlights the need to consider other metrics, such as cash flow and return on investment, to gain a more comprehensive understanding of the company's financial performance.",
    "Among operations, investing, and financing activities, which brought in the most (or lost the least) cash flow for AMD in FY22?": "Based on the provided information, AMD's cash flow for the year 2022 can be analyzed across three main categories: operations, investing, and financing activities.\n\nAccording to the data, AMD's cash flow from operations in 2022 was $3,565 million, which is the highest among the three categories. This indicates that the company generated the most cash from its core business activities, such as generating net income and adjusting for various expenses.\n\nIn contrast, the cash flow from investing activities was $1,999 million, which is significantly lower than the cash flow from operations. This suggests that AMD's investments, such as purchasing property and equipment, short-term investments, and acquisitions, resulted in a net outflow of cash.\n\nOn the other hand, the cash flow from financing activities was -$3,264 million, which means that the company incurred a net outflow of cash from financing activities. This is largely due to the repayment of debt and repurchases of common stock.\n\nTherefore, based on the analysis, it can be concluded that AMD's operations brought in the most cash flow in 2022, followed by a net outflow of cash from investing activities, and a significant net outflow of cash from financing activities.",
    "From FY21 to FY22, excluding Embedded, in which AMD reporting segment did sales proportionally increase the most?": "From FY21 to FY22, excluding the Embedded segment, which AMD reporting segment experienced the most proportional increase in sales?\n\nAccording to the provided information, AMD's Data Center segment saw a significant increase in sales, with net revenue rising by 64% from $3.7 billion in FY21 to $6 billion in FY22. This substantial growth was primarily driven by higher sales of their EPYC server processors.\n\nIn contrast, the Client segment experienced a 10% decrease in net revenue, primarily due to a 24% decline in unit shipments, despite a 19% increase in average selling price. The Gaming segment also saw a decrease in net revenue, with a 4% drop from $5.6 billion in FY21 to $5.6 billion in FY22.\n\nTherefore, the Data Center segment experienced the most proportional increase in sales, with a 64% rise in net revenue, excluding the Embedded segment.",
    "Did AMD report customer concentration in FY22?": "Based on the provided information, AMD did report customer concentration in FY22. Specifically, one customer accounted for 16% of the company's consolidated net revenue for the year ended December 31, 2022. This customer was primarily involved in the Gaming segment, and a loss of this customer would have had a material adverse effect on AMD's business.",
}


class L3bFinanceBenchGenerateDataset(CartridgeGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        doc_names: Optional[List[str]] = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        df = load_finance(doc_names=self.config.doc_names)

        assert config.doc_names == ["AMD_2022_10K"]

        questions: List[FinanceBenchQuestion] = [
            FinanceBenchQuestion(**row) for _, row in df.iterrows()
        ]

        assert len(questions) == 7
        for q in questions:
            assert q.question in L3b_qa_pairs
            q.answer = L3b_qa_pairs[q.question]

        random.Random(42).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]

        self.questions = questions
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> CartridgeGenerateDatasetElement:
        question: FinanceBenchQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=question.question),
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.answer,
            convo_id=question.financebench_id,
            metadata={"idx": index},
        )

    def __len__(self):
        return len(self.questions)


class L3bFinanceBenchEvalDataset(CartridgeDataset):
    class Config(CartridgeDataset.Config):
        _pass_as_config = True
        doc_names: Optional[List[str]] = None

        max_questions: Optional[int] = None
        cot: bool = True

        # ignored
        label_type: str = "tokens"
        data_sources: List[str] = []

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        assert config.doc_names == ["AMD_2022_10K"]
        df = load_finance(doc_names=self.config.doc_names)

        questions: List[FinanceBenchQuestion] = [
            FinanceBenchQuestion(**row) for _, row in df.iterrows()
        ]

        assert len(questions) == 7
        for q in questions:
            assert q.question in L3b_qa_pairs
            q.answer = L3b_qa_pairs[q.question]

        random.Random(42).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]

        if self.config.cot:
            template = COT_PROMPT_TEMPLATE

            def get_answer(question: FinanceBenchQuestion) -> str:
                if question.justification is None:
                    return f"<answer>{question.answer}</answer>"
                else:
                    return (
                        f"{question.justification}\n<answer>{question.answer}</answer>"
                    )

        else:
            template = PROMPT_TEMPLATE

            def get_answer(question: FinanceBenchQuestion) -> str:
                return question.answer

        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=template.format(question=question.question),
                    ),
                    Message(role="assistant", content=get_answer(question)),
                ],
                type="FinanceEval",
                metadata={
                    "question_id": question.financebench_id,
                },
            )
            for question in questions
        ]

        self.tokenizer = tokenizer


