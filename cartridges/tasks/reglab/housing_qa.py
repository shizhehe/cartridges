
import os
import pandas as pd
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from capsules.clients.base import Client, ClientConfig, Sample
from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.structs import Context, ContextConvo, Message, Section
from capsules.generate.run import BaseContextConfig


@dataclass
class QuestionData:
    question: str
    sample: Optional[Sample]
    metadata: Dict[str, Any]
    chunk: Optional[str]=None

class HousingStatutesContextConfig(BaseContextConfig):

    states: list[str] = None
    
    # if None, include all statutes
    num_extra_statutes: Optional[int] = 0

    split_on: Literal["statute", "none"] = "none"

    def instantiate(self) -> Context:
        # SE(03/31): this should load the statutes for the given states 
        # into a `Context` object, placing each statute into a separate `Document`.
        # Ideally, would be nice to filter down to the statutes that are relevant to 
        # housing. We can probably do this by looking at what statutes are cited as 
        # evidence in the `housing_qa` dataset. We should ask Neel if there is a better
        # way to do this.
        from datasets import load_dataset

        statutes = load_dataset("reglab/housing_qa", "statutes", split="corpus", trust_remote_code=True).to_pandas()
        questions = load_dataset("reglab/housing_qa", "questions", split="test", download_mode="force_redownload", trust_remote_code=True).to_pandas()


        sections = []
        for state in self.states: 
            state_statutes = statutes.loc[statutes['state']==state.lower()].copy()

            # add a column consisting of the combined text for each statute
            print(state, len(state_statutes))
            state_statutes['cit_and_text'] = state_statutes[['citation', 'text']].agg(': '.join, axis=1)

            # get the statutes for this state that were cited in the questions
            state_questions = questions[questions["state"] == state] 
            print("num questions: ", len(state_questions))
            relevant_statutes = set(state_questions.explode("statutes")["statutes"].apply(lambda x: x["statute_idx"]))
            housing_statute_df = state_statutes[state_statutes["idx"].isin(relevant_statutes)]

            # also randomly select additional statutes not in relevant_statutes so that we have
            # num_statutes total statutes
            if self.num_extra_statutes is not None and len(housing_statute_df) < self.num_extra_statutes:
                print("relevant statutes: ", len(housing_statute_df))
                additional_statute_df = state_statutes[~state_statutes["idx"].isin(relevant_statutes)]
                additional_statute_df = additional_statute_df.sample(self.num_extra_statutes, random_state=42)
                housing_statute_df = pd.concat([housing_statute_df, additional_statute_df])

            if self.split_on == "none":
                section = Section(
                    desc=f"{state}_statutes",
                    path="reglab/housing_qa/{state}",
                    content="\n\n".join(housing_statute_df['cit_and_text'])
                )
                sections.append(section)
            else:
                # split the statutes into separate sections
                for idx, row in housing_statute_df.iterrows():
                    section = Section(
                        desc=row["path"],
                        path=f"reglab/housing_qa/{state}/{idx}",
                        content=row['cit_and_text']
                    )
                    sections.append(section)
        
        return Context(
            title='state_statutes',
            sections=sections,
        )



class HousingEvalAnswerGenerator(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):
        
        # if none, include all states
        states: list[str] = None
        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        num_top_logprobs: int = 20


    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        from datasets import load_dataset
        self.questions = load_dataset("reglab/housing_qa", "questions", split="test", download_mode="force_redownload", trust_remote_code=True).to_pandas()
        # filter by state!
        self.questions = self.questions[self.questions["state"].isin(config.states)]
        self.state_to_idx = {j:i for i, j in enumerate(config.states)}
        self.answer_client: Client = config.answer_client.instantiate()

    
    def answer_system_prompt_generator(self, question_data: QuestionData) -> str:
        state = question_data.metadata['state']

        return f"""You are a legal expert on the state statutes of {state}."""
    
    def answer_user_prompt_generator(self, context: Context, question_data: QuestionData) -> str:
        """This should generate a prompt asking the model to generate an answer/explanation to the question using 
        the relevant statues and the ground truth answer as context.
        Relevant statutes should be included with both question
        """
        citation = question_data.metadata['citations']
        excerpts = question_data.metadata['excerpts']
        cit_and_excerpts = "\n".join([f"{cit}: {excerpt}" for cit, excerpt in zip(citation, excerpts)])

        # (2) Get the ground truth answer
        answer = question_data.metadata['answer']
        state = question_data.metadata['state']

        return f"""You are a legal expert tasked with answering the following question about {state} state law, using the provided legal statutes as your only source of legal authority.

The correct answer is: {answer}.

Below are excerpts from {state} state statutes that may contain information relevant to the question:
{cit_and_excerpts}

Please generate a clear and concise response that begins with "{answer}", followed by a brief explanation of why this is the correct answer. Your explanation must be grounded in the specific statute(s) that support your answer.  
Only cite statutes that directly and explicitly support your conclusion. If the statutes do not clearly mention the topic asked about in the question, acknowledge that explicitly in your explanation and refrain from making assumptions beyond what the law states.

Keep your response to a few sentences. Focus on legal accuracy, clarity, and relevance.

The question is:  
{question_data.question}"""
    
    def get_questions(self, start_idx: int, end_idx: int) -> list[QuestionData]:
        if start_idx >= len(self.questions):
            raise ValueError(f"start_idx {start_idx} is out of bounds for questions with length {len(self.questions)}")
        if end_idx > len(self.questions):
            raise ValueError(f"end_idx {end_idx} is out of bounds for questions with length {len(self.questions)}")
        
        chosen_rows = self.questions.iloc[start_idx:end_idx]
        # for row in chosen_rows.itertuples(index=False):
        #     print("question: ", row.question)
        #     print("true answer: ", row.answer)
        #     print("citations: ", [stat['citation'] for stat in row.statutes])
  
        return [
            QuestionData(
                question=row.question,
                sample=row.answer,
                metadata={
                    "state": row.state,
                    "idx": row.idx,
                    "answer": row.answer,
                    "citations": [stat['citation'] for stat in row.statutes],
                    "excerpts": [stat['excerpt'] for stat in row.statutes],
                },
            )
            for row in chosen_rows.itertuples(index=False)
        ]


    def sample_convos(self, batch_idx: int, num_convos: int) -> list[ContextConvo]:
        # This should generate answers to questions from the dataset in the range
        # [start_idx, end_idx) by prompting the `answer_client` with the question,
        # the cited excerpts, and the ground truth answer.
        # (1) Sample questions
        start_idx = (batch_idx * num_convos)
        end_idx = start_idx + num_convos

        questions = self.get_questions(start_idx, end_idx)
        try:
            assert len(questions) == num_convos
        except:
            breakpoint()

        # (2) Sample answers
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": self.answer_system_prompt_generator(question_data),
                },
                {
                    "role": "user",
                    "content": self.answer_user_prompt_generator(self.context, question_data),
                },
            ]
            for question_data in questions
        ]

        answer_samples = self.answer_client.chat(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
        ).samples

        # (3) Construct convos
        convos = [
            ContextConvo(
                messages=[
                    Message(
                        content=question.question,
                        role="user",
                        # no sample here because question was provided by the dataset
                    ),
                    Message(
                        content=answer_sample.text,
                        role="assistant",
                        sample=answer_sample,
                    ),
                ],
                type=self.__class__.__name__,
                metadata=question.metadata,
                id=str(uuid.uuid4()),  # SE(03/10): Added this to make sure every sample has a unique id
            )
            for question, answer_sample in zip(questions, answer_samples, strict=True)
        ]

        return convos
    
    def sample_convo(self, idx: int) -> ContextConvo:
        # This should generate an answer to a single question from the dataset
        # at index `idx`.

        raise NotImplementedError()

    def __len__(self) -> int:
        # SE(04/01): this should return the number of questions in the dataset.
        # and is used by the generate script to determine the number of batches
        # to launch.
        return len(self.questions)
    

class EvictionCategoriesMCGenerator(ContextConvoGenerator): 

    class Config(ContextConvoGenerator.Config):
        
        # if none, include all states
        states: list[str] = None
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        num_top_logprobs: int = 20


    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        from datasets import load_dataset
        self.questions = pd.read_json(
            os.path.join(
                os.environ.get("CAPSULES_DIR", "./"), 
                "capsules/tasks/reglab/housing_qa_consolidated_generations_with_answer_options.jsonl", 
            ),
            lines=True
        )
        # filter by state!
        self.questions = self.questions[self.questions["state"].isin(config.states)]
        # filter by question type
        self.questions = self.questions[self.questions['category'] == 'eviction_categories']

    
    def question_text_generator(self, question_data: QuestionData) -> str: 
        return f"""In the state of {question_data.metadata['state']}, {question_data.question.lower()} Select all of the following options that apply:
{question_data.metadata['answer_options']}
Give your answer in a newline separated list, with no other additional text."""
    

    def sample_convos(self, batch_idx: int, num_convos: int) -> list[ContextConvo]:
        # This should generate answers to questions from the dataset in the range
        # [start_idx, end_idx)

        # Select rows of data
        start_idx = (batch_idx * num_convos)
        end_idx = start_idx + num_convos

        if start_idx >= len(self.questions):
            raise ValueError(f"start_idx {start_idx} is out of bounds for questions with length {len(self.questions)}")
        if end_idx > len(self.questions):
            raise ValueError(f"end_idx {end_idx} is out of bounds for questions with length {len(self.questions)}")

        

        # Construct question text
        chosen_rows = self.questions.iloc[start_idx:end_idx]
        questions = [
            QuestionData(
                question=row.question,
                sample=None,
                metadata={
                    "state": row.state,
                    "answer_options": row.answer_options,
                },
            )
            for row in chosen_rows.itertuples(index=False)
        ]
        answers = chosen_rows['answer'].tolist()

        try:
            assert len(questions) == num_convos
        except:
            breakpoint()    

        # (3) Construct convos
        convos = [
            ContextConvo(
                messages=[
                    Message(
                        content=self.question_text_generator(question),
                        role="user",
                        # no sample here because question was provided by the dataset
                    ),
                    Message(
                        content=answer,
                        role="assistant",
                    ),
                ],
                type=self.__class__.__name__,
                metadata=question.metadata,
                id=str(uuid.uuid4()),  # SE(03/10): Added this to make sure every sample has a unique id
            )
            for question, answer in zip(questions, answers, strict=True)
        ]

        return convos

    
    def __len__(self) -> int:
        return len(self.questions)
    

class EvictionTemporalMiscEvalGenerator(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):
        
        # if none, include all states
        states: list[str] = None
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        num_top_logprobs: int = 20


    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        from datasets import load_dataset
        self.questions = pd.read_json("capsules/tasks/reglab/housing_qa_consolidated_generations_with_answer_options.jsonl", lines=True)
        # filter by state!
        self.questions = self.questions[self.questions["state"].isin(config.states)]
        self.questions = self.questions[
            (self.questions['category'] == 'eviction_temporal') | 
            (self.questions['category'] == 'eviction_misc')
        ]

    def sample_convos(self, batch_idx: int, num_convos: int) -> list[ContextConvo]:
        # This should generate answers to questions from the dataset in the range
        # [start_idx, end_idx)

        # Select rows of data
        start_idx = (batch_idx * num_convos)
        end_idx = start_idx + num_convos

        if start_idx >= len(self.questions):
            raise ValueError(f"start_idx {start_idx} is out of bounds for questions with length {len(self.questions)}")
        if end_idx > len(self.questions):
            raise ValueError(f"end_idx {end_idx} is out of bounds for questions with length {len(self.questions)}")

        # Construct question text
        chosen_rows = self.questions.iloc[start_idx:end_idx]
        questions = [
            QuestionData(
                question=row.question,
                sample=None,
                metadata={
                    "state": row.state,
                    "category": row.category,
                    "statutes": row.statutes,
                },
            )
            for row in chosen_rows.itertuples(index=False)
        ]
        answers = chosen_rows['answer'].tolist()

        try:
            assert len(questions) == num_convos
        except:
            breakpoint()    

        # (3) Construct convos
        convos = [
            ContextConvo(
                messages=[
                    Message(
                        content=question.question,
                        role="user",
                        # no sample here because question was provided by the dataset
                    ),
                    Message(
                        content=answer,
                        role="assistant",
                    ),
                ],
                type=self.__class__.__name__,
                metadata=question.metadata,
                id=str(uuid.uuid4()),  # SE(03/10): Added this to make sure every sample has a unique id
            )
            for question, answer in zip(questions, answers, strict=True)
        ]

        return convos

    
    def __len__(self) -> int:
        return len(self.questions)
    
class StatuteRecallEvalGenerator(ContextConvoGenerator): 

    class Config(ContextConvoGenerator.Config):
        
        # if none, include all states
        states: list[str] = None
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        num_top_logprobs: int = 20


    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        from datasets import load_dataset
        self.questions = pd.read_json("capsules/tasks/reglab/housing_qa_consolidated_generations_with_answer_options.jsonl", lines=True)
        # filter by state!
        self.questions = self.questions[self.questions["state"].isin(config.states)]


    def question_text_generator(self, question_data: QuestionData) -> str: 
        return f"""Which statutes, if any in the {question_data.metadata['state']} state legal code are relevant to answering the following question?
{question_data.question} 
Give your answer in the form of a list of statute citations, followed by the full exact text of the statute, as in the example below. If there are no relevant statutes, say "No relevant statutes". Do not include any additional text.
<example>
ALA. CODE \u00a7 35-9A-501(B)
(b) If a landlord acts in violation of subsection (a), the tenant is entitled to the remedies provided in Section 35-9A-407 and has a defense in any retaliatory action against the tenant for possession.
<\example>"""


    def sample_convos(self, batch_idx: int, num_convos: int) -> list[ContextConvo]:
        # This should generate answers to questions from the dataset in the range
        # [start_idx, end_idx)
        # EL TODO: finish writing this

        # Select rows of data
        start_idx = (batch_idx * num_convos)
        end_idx = start_idx + num_convos

        if start_idx >= len(self.questions):
            raise ValueError(f"start_idx {start_idx} is out of bounds for questions with length {len(self.questions)}")
        if end_idx > len(self.questions):
            raise ValueError(f"end_idx {end_idx} is out of bounds for questions with length {len(self.questions)}")

        # Construct question text
        chosen_rows = self.questions.iloc[start_idx:end_idx]
        questions = [
            QuestionData(
                question=row.question,
                sample=None,
                metadata={
                    "state": row.state,
                    "category": row.category,
                    "statutes": row.statutes,
                },
            )
            for row in chosen_rows.itertuples(index=False)
        ]
        answers = ["\n\n".join([stat[0] + ":\n" + stat[1] for stat in row.statutes]) 
                   if len(row.statutes) > 0 else "No relevant statutes" 
                   for row in chosen_rows.itertuples(index=False)]

        try:
            assert len(questions) == num_convos
        except:
            breakpoint()    

        # (3) Construct convos
        convos = [
            ContextConvo(
                messages=[
                    Message(
                        content=self.question_text_generator(question),
                        role="user",
                        # no sample here because question was provided by the dataset
                    ),
                    Message(
                        content=answer,
                        role="assistant",
                    ),
                ],
                type=self.__class__.__name__,
                metadata=question.metadata,
                id=str(uuid.uuid4()),  # SE(03/10): Added this to make sure every sample has a unique id
            )
            for question, answer in zip(questions, answers, strict=True)
        ]

        return convos

