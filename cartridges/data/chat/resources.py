import json
from dataclasses import dataclass
import random
from typing import Dict, List, Literal

import aiofiles
from cartridges.data.resources import Resource


@dataclass 
class Message: 
    message_id: str
    role: str
    content: str
    create_time: float
    update_time: float

@dataclass 
class Conversation: 
    conversation_id: str
    title: str
    create_time: float
    update_time: float

    messages: List[Message]

def _parse_openai_chat_history(data: List[Dict]) -> List[Conversation]:
    conversations = []
    for convo in data:
        messages = []
        for msg in list(convo["mapping"].values()):
            if msg["message"] is None:
                continue
            msg = msg["message"]
            
            if msg["content"]["content_type"] != "text":
                continue

            # [SE 08/24] Nowhere in my history is parts ever more than length 1
            content = "".join(msg["content"]["parts"])
            if content == "":
                continue
            msg = Message(
                message_id=msg["id"],
                role=msg["author"]["role"],
                content=content,
                create_time=msg["create_time"],
                update_time=msg["update_time"]
            )
            messages.append(msg)

        convo = Conversation(
            conversation_id=convo["conversation_id"],
            title=convo["title"],
            create_time=convo["create_time"],
            update_time=convo["update_time"],
            messages=messages
        )
        conversations.append(convo)
    return conversations



class ChatHistoryResource(Resource):

    class Config(Resource.Config):
        path: str
        provider: Literal["openai"] = "openai"

    def __init__(self, config: Config):
        self.config = config
        self.conversations: List[Conversation] | None = None
    
    async def setup(self):
        async with aiofiles.open(self.config.path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        if self.config.provider == "openai":
            self.conversations = _parse_openai_chat_history(data)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if not self.conversations or len(self.conversations) == 0:
            raise ValueError("No conversations available. Make sure to call setup() first.")

        # Randomly sample a conversation
        conversation = random.choice(self.conversations)

        context = "\n\n".join([f"--begin {msg.role} [created at {msg.create_time}] message---\n{msg.content}\n--end {msg.role} message---" for msg in conversation.messages])    
        
        seed_prompts = ["Ask a question about the content above."] * batch_size
        return context, seed_prompts



        

