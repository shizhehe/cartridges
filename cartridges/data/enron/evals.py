from typing import List, Optional, Tuple, Dict
import json
import re
import random
import os

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement, DataSource
from cartridges.models.helpers import ModelHelper
from cartridges.clients.openai import OpenAIClient


class EnronQAGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        judge_model: str = "gpt-4o-mini"  # LLM judge model
        use_llm_judge: bool = True  # Whether to use LLM judge or fallback to string matching
        judge_temperature: float = 0.0  # Temperature for judge model
        judge_max_tokens: int = 300  # Max tokens for judge response

    def __init__(self, config: Config, model_helper: ModelHelper, seed: int):
        # Call parent constructor to load data from DataSource
        super().__init__(config, model_helper, seed)
        
        # Initialize Cartridges OpenAI client for LLM judging
        if self.config.use_llm_judge:
            self.judge_client = OpenAIClient.Config(
                model_name=self.config.judge_model,
                api_key=os.getenv("OPENAI_API_KEY")
            ).instantiate()
        
        # Convert loaded conversations to QA format for easier access
        self.qa_items = []
        for i, conversation in enumerate(self.data):
            if len(conversation.messages) >= 2:
                question = ""
                answer = ""
                
                # Extract question and answer from conversation messages
                for msg in conversation.messages:
                    if msg.role == "user":
                        question = msg.content
                    elif msg.role == "assistant":
                        answer = msg.content
                
                if question and answer:
                    self.qa_items.append({
                        'question': question,
                        'answer': answer,
                        'id': f"qa_{i}",
                        'conversation_idx': i
                    })

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        qa_item = self.qa_items[index]
        
        # Format the question as a proper prompt
        prompt = f"Question: {qa_item['question']}\nAnswer:"
        
        kwargs = self.model_helper.get_apply_chat_template_kwargs(False)  # No CoT for now
        
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=self.model_helper.get_chat_template(),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=qa_item['answer'],
            convo_id=qa_item['id'],
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.qa_items)

    def _llm_judge_score(self, pred: str, answer: str, question: str) -> Tuple[bool, Dict[str, Optional[str]]]:
        """Use LLM as a judge to score the prediction against the correct answer."""
        
        judge_prompt = f"""You are evaluating whether a model's answer to a question is correct or equivalent to the reference answer.

Question: {question}

Reference Answer: {answer}

Model's Answer: {pred}

Task: Determine if the model's answer is semantically equivalent to the reference answer, even if worded differently. Consider:
1. Do they convey the same meaning?
2. Are the key facts/information the same?
3. Is the model's answer a reasonable paraphrase or reformulation?
4. For factual questions, do they provide the same factual information?

You must respond with a valid JSON object with the following format:
{{
    "judgment": "CORRECT" | "INCORRECT",
    "explanation": "Your brief explanation (1-2 sentences)"
}}

Where:
- "CORRECT" if the answers are semantically equivalent
- "INCORRECT" if they are not equivalent or the model's answer is wrong

Example:
{{"judgment": "CORRECT", "explanation": "The model's answer provides the same factual information as the reference answer."}}"""

        try:
            # Use Cartridges OpenAI client for judging
            import asyncio
            
            async def get_judge_response():
                return await self.judge_client.chat(
                    chats=[[
                        {"role": "system", "content": "You are an expert evaluator for question-answering tasks. Always respond with valid JSON."},
                        {"role": "user", "content": judge_prompt}
                    ]],
                    temperature=self.config.judge_temperature,
                    max_completion_tokens=self.config.judge_max_tokens
                )
            
            # Run the async function
            chat_response = asyncio.run(get_judge_response())
            judge_response = chat_response.samples[0].text.strip()
            
            # Parse the JSON response
            try:
                judge_data = json.loads(judge_response)
                
                judgment = judge_data.get("judgment", "INCORRECT").upper()
                explanation = judge_data.get("explanation", "No explanation provided")
                
                # Validate judgment - only accept CORRECT or INCORRECT
                if judgment not in ["CORRECT", "INCORRECT"]:
                    judgment = "INCORRECT"  # Default to incorrect for safety
                
                # Determine if correct
                is_correct = judgment == "CORRECT"
                
                return is_correct, {
                    "match_type": "llm_judge",
                    "judgment": judgment,
                    "explanation": explanation,
                    "judge_model": self.config.judge_model,
                    "raw_response": judge_response
                }
                
            except json.JSONDecodeError as json_error:
                print(f"Failed to parse LLM judge JSON response: {json_error}")
                print(f"Raw response: {judge_response}")
                # Try to extract judgment from raw text as fallback
                judge_response_upper = judge_response.upper()
                if "CORRECT" in judge_response_upper and "INCORRECT" not in judge_response_upper:
                    return True, {
                        "match_type": "llm_judge_text_fallback",
                        "judgment": "CORRECT",
                        "explanation": "Extracted from non-JSON response",
                        "judge_model": self.config.judge_model,
                        "raw_response": judge_response
                    }
                else:
                    # Final fallback to string matching
                    return self._fallback_score(pred, answer)
            
        except Exception as e:
            print(f"Error in LLM judge scoring: {e}")
            # Fallback to string matching
            return self._fallback_score(pred, answer)
    
    def _fallback_score(self, pred: str, answer: str) -> Tuple[bool, Dict[str, Optional[str]]]:
        """Fallback scoring using string matching when LLM judge fails."""
        
        # Clean up the strings
        pred_clean = pred.strip().lower()
        answer_clean = answer.strip().lower()
        
        # Exact match
        if pred_clean == answer_clean:
            return True, {"match_type": "exact_fallback", "pred_clean": pred_clean}
        
        # Substring match (answer in prediction)
        if answer_clean in pred_clean:
            return True, {"match_type": "substring_fallback", "pred_clean": pred_clean}
        
        # Reverse substring match (prediction in answer, for short predictions)
        if len(pred_clean) > 5 and pred_clean in answer_clean:
            return True, {"match_type": "reverse_substring_fallback", "pred_clean": pred_clean}
        
        # Token overlap similarity (for more flexible matching)
        pred_tokens = set(pred_clean.split())
        answer_tokens = set(answer_clean.split())
        
        if len(answer_tokens) > 0:
            overlap = len(pred_tokens & answer_tokens)
            similarity = overlap / len(answer_tokens)
            
            # Consider it correct if significant overlap (>= 0.6)
            if similarity >= 0.6:
                return True, {
                    "match_type": "token_overlap_fallback", 
                    "pred_clean": pred_clean,
                    "similarity": similarity
                }
        
        # No match found
        return False, {
            "match_type": "no_match_fallback", 
            "pred_clean": pred_clean,
            "similarity": 0.0
        }

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        """
        Score the prediction against the correct answer for Enron QA tasks.
        
        Uses LLM judge (GPT-4o-mini) for semantic evaluation, with fallback to string matching.
        """
        
        # Get the question for context in LLM judging
        question = ""
        if hasattr(self, 'qa_items') and self.qa_items:
            # Extract question from convo_id if possible
            try:
                qa_idx = int(convo_id.split('_')[-1]) if '_' in convo_id else 0
                if 0 <= qa_idx < len(self.qa_items):
                    question = self.qa_items[qa_idx]['question']
            except (ValueError, IndexError):
                question = "Question context not available"
        
        # Use LLM judge if enabled and available
        if self.config.use_llm_judge and hasattr(self, 'judge_client'):
            return self._llm_judge_score(pred, answer, question)
        else:
            # Fallback to string matching
            return self._fallback_score(pred, answer)