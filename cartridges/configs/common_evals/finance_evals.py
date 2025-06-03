from dataclasses import dataclass


########## SYSTEM PROMPT ##########


QUESTION_SYSTEM_PROMPT_TEMPLATE = """Your job is to generate questions that are answered by using information in the following financial statement.
Your question will be given to someone to test their understanding of the financial statement.

<title>
{title}
</title>

<document>
{content}
</document>
"""


ANSWER_SYSTEM_PROMPT_TEMPLATE = """Please use the information in the following financial document to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
"""


########## Factual and knowledge based questions (using localized information within the document) #########

# FACTUAL TRACES
QUESTION_PROMPT_TEMPLATE_FACTUAL = """Please generate a question to test someone's ability to remember factual details from the document. The answer should be a few tokens long and be a factual detail from the statement, such as a number, entity, date, title, or name.

This question should not be common knowledge: instead, it should be something that is only answerable via information in the document.
"""

# KNOWLEDGE
QUESTION_PROMPT_TEMPLATE_KNOWLEDGE = """Please generate a question that requires combining information mentioned both inside and outside the document. 

This question should require using a fact from the document and also a fact that you are confident about, but is not mentioned in the document. For instance: 
- What are the founding dates of the companies that got acquired this year? This is a good question because the names of the acquired companies are mentioned in the document and the founding dates are not mentioned.
- What is the name of the CEO's spouse?  This is a good question because the name of the CEO is mentioned in the document and the spouse's name is not mentioned. 

The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name."""

# DISJOINT TRACES
QUESTION_PROMPT_TEMPLATE_DISJOINT = """Please generate a multi-hop question that tests someone's ability to use factual information mentioned in at least two very different sub-sections of the document. 

This question shouldn't be a standard question about this kind of document. Instead, it should ask about two particularly disconnected ideas, like comparing information about the amount of owned space for the company headquarters with the amount of dollars of estimated liability or comparing the revenue number with the number of employees.

This question should also test one's ability to do retrieval: do not give away part of the answer in the question. Ensure that for one to get the correct answer to the question, they need to understand the document.

The answer should be a short: for example, a number, entity, date, title, or name."""


######### Document comprehension (cross-document information) ###########

# SYNTHESIZE TRACES (summarize, outline -- requires full document)
QUESTION_PROMPT_TEMPLATE_SYNTHESIZE = """Please generate a question that requires synthesizing and aggregating information in the document. 

For instance, you could ask someone to summarize a page of the document, list all the key competitors mentioned in the document, or summarize the company's business model."""

# STRUCTURE TRACES
QUESTION_PROMPT_TEMPLATE_STRUCTURE = """Please generate a question that requires understanding the structure of the document. 

This question should be more about the structure of the document, rather than the precise statement details. For instance, you could ask someone to list the titles of all the sections in the document, describe the document structure, report the total number of pages, ask which section amongst two sections comes first, or report the section with the largest number of tables.
"""

# CREATIVE TRACES (unexpected)
QUESTION_PROMPT_TEMPLATE_CREATIVE = """Please generate a question about the document to test someone's ability to comprehend the content of the document. This question specifically should be focused on their ability to generalize the information about the document to a strange question of sorts.

This question shouldn't be a standard question about this kind of document, it should ask to do something abnormal and creative, like writing a poem about a financial document."""


########## Algorithmic reasoning (reasoning beyond information that's been directly provided to the model) #########

# COUNTING TRACES
QUESTION_PROMPT_TEMPLATE_COUNTING = """Please generate a question that requires counting how frequently different events occur in the document.

This question should be about statistical properties of the document, rather than the statement details. For instance, you could ask someone to count the number of times the word "million" is mentioned or count the length of the shortest section title.

The answer should be a number."""

# REASOINING TRACES
QUESTION_PROMPT_TEMPLATE_REASONING = """Please generate a question that requires mathematical reasoning over the values in the document. 

This question should require going beyond the facts directly mentioned in the statement, such as asking to compute the percentage increase in revenue between two years, find the largest expense category, or calculate difference in profit between two years. 

The answer should be a number."""

# CODING TRACES
QUESTION_PROMPT_TEMPLATE_CODING = """Please generate a question that requires someone to write a short Python program to analyze the document. 

This question should be about statistical properties of the document, rather than the statement details. For instance, you could ask someone to write a program that returns the number of times the word "million" is mentioned or that returns the word mentioned more times between "million" and "thousand". 
However, one shouldn't be able to answer the question without knowledge of the document.

The answer should be a program that would return a few tokens or a number. Only provide the imports and function "def parse(document)" that takes the document string as input. Do not provide any other explanation."""

QUESTION_PROMPT_TEMPLATE_MEMORIZATION = """Your primary objective is to generate questions designed specifically to test how well a user *verbatim memorized* the provided document. Your questions should demand the precise, word-for-word reproduction of specific text blocks, simulating a rigorous memorization test.

To construct such a question, you must first pinpoint an exact starting location within the document. This is achieved by selecting a short, unique phrase (approximately 5-15 words) from the document which will serve as an anchor for the question. This anchor phrase *must be directly quoted within the question itself* to clearly indicate the starting point for recall. Following the quoted anchor, your question needs to instruct the respondent to write out the subsequent text from the original document, beginning immediately after the anchor ends. In your question, you should also provide the rough location of the text in the document. You should also provide guidance on the expected length of the recalled passage, for instance, by specifying "the next 250 words" or "the following three paragraphs". You should ask for approximately 300 words, depending on the context

Use unambiguous phrasing in your generated question. The generated question must leave absolutely no room for interpretation regarding the starting point or the verbatim nature of the required answer."""


ANSWER_PROMPT_TEMPLATE_OPEN_ENDED = """{question}"""

ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER = """{question}.

Please think through the problem first. Then, output your final answer between <answer> and </answer> tags."""

ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER_NUMERIC = (
    f"""{ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER}. The answer should be a number."""
)

ANSWER_PROMPT_TEMPLATE_CODING = f"""{ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER}. The answer should be a program that would return a short answer or a number. Only provide the imports and a function "def parse(document)", which takes the document string as input."""


@dataclass
class FinanceEvalMetadata:
    tag: str
    question_prompt_template: str
    answer_prompt_template: str

    def get_artifact_name(
        self,
        doc_name: str,
        num_samples: int,
        version_tag: str,
        model_type: str,
    ):
        return f"finance_eval_doc_{doc_name}_tag_{self.tag}_samples{num_samples}_{version_tag}_model_{model_type}"


FinanceEvals = [
    # document fact localization
    FinanceEvalMetadata(
        "factual",
        QUESTION_PROMPT_TEMPLATE_FACTUAL,
        ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER,
    ),
    # FinanceEvalMetadata(
    #     "knowledge",
    #     QUESTION_PROMPT_TEMPLATE_KNOWLEDGE,
    #     ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER,
    # ),
    FinanceEvalMetadata(
        "disjoint",
        QUESTION_PROMPT_TEMPLATE_DISJOINT,
        ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER,
    ),
    # document comprehension overall
    FinanceEvalMetadata(
        "synthesize",
        QUESTION_PROMPT_TEMPLATE_SYNTHESIZE,
        ANSWER_PROMPT_TEMPLATE_OPEN_ENDED,
    ),
    FinanceEvalMetadata(
        "structure",
        QUESTION_PROMPT_TEMPLATE_STRUCTURE,
        ANSWER_PROMPT_TEMPLATE_OPEN_ENDED,
    ),
    FinanceEvalMetadata(
        "creative", QUESTION_PROMPT_TEMPLATE_CREATIVE, ANSWER_PROMPT_TEMPLATE_OPEN_ENDED
    ),
    # algorithmic reasoning
    FinanceEvalMetadata(
        "counting",
        QUESTION_PROMPT_TEMPLATE_COUNTING,
        ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER_NUMERIC,
    ),
    FinanceEvalMetadata(
        "reasoning",
        QUESTION_PROMPT_TEMPLATE_REASONING,
        ANSWER_PROMPT_TEMPLATE_SPECIFIC_ANSWER_NUMERIC,
    ),
    FinanceEvalMetadata(
        "coding", QUESTION_PROMPT_TEMPLATE_CODING, ANSWER_PROMPT_TEMPLATE_CODING
    ),
    FinanceEvalMetadata(
        "memorization",
        QUESTION_PROMPT_TEMPLATE_MEMORIZATION,
        ANSWER_PROMPT_TEMPLATE_OPEN_ENDED,
    ),
]
