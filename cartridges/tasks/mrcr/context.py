from __future__ import annotations
from datetime import datetime, timedelta
import hashlib
import json
import os
import random
from typing import List
import uuid
from requests import session

import numpy as np
import pandas as pd
from datasets import load_dataset

from capsules.context import StructuredContext
from capsules.generate.run import BaseContextConfig
from capsules.tasks.mrcr import MRCRQuestion
from capsules.utils import get_logger

logger = get_logger(__name__)

def get_message_df(
    num_dataset_rows: int = 32,
    seed: int = 42,
    num_user_messages: int = 50,
) -> pd.DataFrame:
    
    dataset = load_dataset("openai/mrcr")['train']

    messages = []
    random.seed(seed)
    rows = random.choices(range(len(dataset)), k=num_dataset_rows)
    for idx in rows:
        messages += json.loads(dataset["prompt"][idx])

    # (1) Create a dataframe with two columns: user_message and assistant_message
    # and drop any duplicates to ensure that the all of the assistant responses
    # are unique.
    results = []
    user_message = None
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
        else:
            assert user_message is not None
            results.append(
                {
                    "user_message": user_message,
                    "assistant_message": message["content"]
                }
            )
    df = pd.DataFrame(results).drop_duplicates()

    # (2) Get user messages with the most occurrences and filter the messages
    # to only include the most common user messages.
    user_messages = (
        df.groupby("user_message").size().sort_values(ascending=False).iloc[:num_user_messages]
    )
    df = df[df["user_message"].isin(user_messages.index)]


    # (3) randomly shuffle 
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # (4) melt the dataframe
    messages = []
    for row in df.to_dict(orient="records"):
        messages.extend(
            [
                {"role": "user", "content": row["user_message"]},
                {"role": "assistant", "content": row["assistant_message"]}
            ]
        )
    df = pd.DataFrame(messages)

    # (5) add random timestamps and ids
    start_date = datetime(2024, 1, 1)
    np.random.seed(seed)
    intervals = np.random.randint(0, 60 * 24, size=len(df))
    intervals = np.cumsum(intervals).tolist()
    df["timestamp"] = pd.Series(
        [start_date + timedelta(minutes=interval) for interval in intervals]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["id"] = df.apply(lambda _: str(uuid.uuid4()).split("-")[0], axis=1)
    return df



class MRCRStructuredContextConfig(BaseContextConfig):
    num_dataset_rows: int = 32
    num_user_messages: int = 40
    seed: int = 42


    def instantiate(self) -> ChatSession:
        hash_key = hashlib.md5(json.dumps(self.to_dict()).encode()).hexdigest()
        cache_path = os.path.join(
            os.environ["CAPSULES_DIR"],
            "output",
            "mrcr",
            f"{hash_key}.feather"
        )
        if os.path.exists(cache_path):
            logger.info(f"Loading message dataframe from cache: {cache_path}")
            df = pd.read_feather(cache_path)
        else:
            logger.info(f"No cache found at {cache_path}, generating new dataframe")
            df = get_message_df(
                num_dataset_rows=self.num_dataset_rows,
                seed=self.seed,
                num_user_messages=self.num_user_messages
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            logger.info(f"Saving message dataframe to cache: {cache_path}")
            df.to_feather(cache_path)
        
        # (6) create sessions
        messages = [
            Message(**message)
            for message in df.to_dict(orient="records")
        ]

        return ChatSession(
            session_id="session_01",
            messages=messages
        )
        



class Message(StructuredContext):
    id: str
    timestamp: str

    role: str
    content: str


class ChatSession(StructuredContext):
    session_id: str

    messages: List[Message]


class MRCRContext(StructuredContext):
    sessions: List[ChatSession]

