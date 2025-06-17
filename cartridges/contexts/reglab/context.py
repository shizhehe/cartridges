from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from seaborn import load_dataset

from cartridges.context import TexDocument, TexChapter, TexSection
from cartridges.context import BaseContextConfig

from cartridges.context import StructuredContext
from cartridges.structs import Section
from cartridges.tasks.longhealth.load import load_longhealth_dataset, LongHealthPatient, LongHealthQuestion

class ReglabHousingStructuredContextConfig(BaseContextConfig):
    states: Optional[list[str]] = None
    # if None, include all statutes
    num_extra_statutes_per_state: Optional[int] = 0
    
    
    def instantiate(self) -> StructuredContext:
        from datasets import load_dataset
        statute_df = load_dataset("reglab/housing_qa", "statutes", split="corpus", trust_remote_code=True).to_pandas()
        question_df = load_dataset("reglab/housing_qa", "questions", split="test", download_mode="force_redownload", trust_remote_code=True).to_pandas()

        if self.states is not None:
            question_df = question_df[question_df["state"].isin(self.states)]

        relevant_statutes = set(question_df.explode("statutes")["statutes"].apply(lambda x: x["statute_idx"]))
        housing_statute_df = statute_df[statute_df["idx"].isin(relevant_statutes)]

        if self.num_extra_statutes_per_state is not None and self.num_extra_statutes_per_state > 0:
            other_statute_df = statute_df[~statute_df["idx"].isin(relevant_statutes)].sample(
                self.num_extra_statutes_per_state * housing_statute_df["state"].nunique(),
                random_state=42,
            )
            housing_statute_df = pd.concat([housing_statute_df, other_statute_df])
        
        state_codes = []
        for state, state_df in housing_statute_df.groupby("state"):
            state_code = StateCode(
                state=state,
                statutes=[
                    Statute(citation=row["citation"], path=row["path"], text=row["text"]) 
                    for _, row in state_df.iterrows()
                ],
            )
            state_codes.append(state_code)

        state_codes_context = StateCodes(
            state_codes=state_codes,
        )
        return state_codes_context


class Statute(StructuredContext):
    citation: str
    path: str
    text: str

class StateCode(StructuredContext):
    state: str 
    statutes: List[Statute]


class StateCodes(StructuredContext):
    state_codes: List[StateCode]