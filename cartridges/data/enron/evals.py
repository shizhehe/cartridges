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
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.sglang_modal import SGLangClient


class EnronQAGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        qa_judge_model: Optional[OpenAIClient.Config] = None
        qa_use_llm_judge: bool = True  # Whether to use LLM judge or fallback to string matching
        qa_judge_temperature: float = 0.0  # Temperature for judge model
        qa_judge_max_tokens: int = 300  # Max tokens for judge response
        perplexity_judge_model: Optional[SGLangClient.Config] = None

    def __init__(self, config: Config, model_helper: ModelHelper, seed: int):
        # Call parent constructor to load data from DataSource
        super().__init__(config, model_helper, seed)
        
        # Initialize Cartridges OpenAI client for LLM judging
        if self.config.qa_use_llm_judge:
            self.qa_judge_client = self.config.qa_judge_model.instantiate()
        
        if self.config.perplexity_judge_model:
            self.perplexity_judge_client = self.config.perplexity_judge_model.instantiate()
        
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
                        'conversation_idx': i,
                        'system_prompt': conversation.system_prompt
                    })

        # baseline perplexity scores, what is log likelihood of ground-truth answer
        self.baselines = []
        baseline_perplexity_scores = []
        for qa_item in self.qa_items:
            perplexity_scores, perplexity_metadata = self._perplexity_judge_score(qa_item['answer'], qa_item['question'], qa_item['system_prompt'])
            baseline_perplexity_scores.append(perplexity_scores['perplexity'])
        self.baselines.append({'perplexity': np.mean(baseline_perplexity_scores)})

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

    def _perplexity_judge_score(self, pred: str, question: str, system_prompt: str) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
        """Use SGLang to calculate perplexity of the prediction given the context."""
        if not hasattr(self, 'perplexity_judge_client'):
            return {"perplexity": None}, {"error": "No perplexity judge client configured"}
        
        try:
            import numpy as np
            
            # Construct the conversation with system prompt, question, and prediction
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": pred}
            ]
            
            # Use SGLangClient to get detailed logprobs including input logprobs
            # Set max_completion_tokens=1 to minimize generation, focus on the logprobs of existing conversation
            response = self.perplexity_judge_client.chat(
                chats=[conversation],
                max_completion_tokens=1,
                temperature=0.0,
                return_logprob=True,
            )
            
            if not response.samples or len(response.samples) == 0:
                return {"perplexity": None}, {"error": "No response samples returned"}
                
            sample = response.samples[0]
            
            # SGLangClient provides input_log_prob which is what we need for perplexity calculation
            if not hasattr(sample, 'input_log_prob') or sample.input_log_prob is None:
                return {"perplexity": None}, {"error": "No input logprobs available"}
            
            # Convert input_log_prob to numpy array if it isn't already
            if isinstance(sample.input_log_prob, (list, tuple)):
                input_logprobs = np.array(sample.input_log_prob)
                tokens = sample.tokens
            else:
                input_logprobs = sample.input_log_prob
                tokens = sample.tokens
            
            # Extract logprobs for only the prediction tokens
            context_conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            context_tokens = self.perplexity_judge_client.tokenizer.apply_chat_template(
                context_conversation,
                add_generation_prompt=True,
                return_tensors=None,
                tokenize=True
            )
            num_context_tokens = len(context_tokens)
            
            # The prediction logprobs start after the context tokens
            # Note: there might be some special tokens for role transitions
            if len(input_logprobs) <= num_context_tokens:
                return {"perplexity": None}, {"error": "No prediction tokens found in logprobs"}
            
            # Extract only the logprobs for the prediction tokens
            prediction_logprobs = input_logprobs[num_context_tokens:]
            prediction_tokens = tokens[num_context_tokens:]
            print(f"Prediction tokens: {prediction_tokens}")
            
            # Filter out any invalid logprobs (NaN, -inf, etc.)
            valid_logprobs = prediction_logprobs[~np.isnan(prediction_logprobs) & ~np.isinf(prediction_logprobs)]
            
            if len(valid_logprobs) == 0:
                return {"perplexity": None}, {"error": "No valid logprobs found"}
            
            # Calculate perplexity: exp(-mean(log_probs))
            mean_logprob = np.mean(valid_logprobs)
            perplexity = np.exp(-mean_logprob)
            
            return {"perplexity": float(perplexity)}, {
                "method": "sglang_prediction_logprobs", 
                "model": self.config.perplexity_judge_model.model_name,
                "num_prediction_tokens": len(valid_logprobs),
                "num_context_tokens": num_context_tokens,
                "mean_logprob": float(mean_logprob)
            }
            
        except Exception as e:
            return {"perplexity": None}, {"error": f"Perplexity calculation failed: {str(e)}"}}

    def _llm_judge_score(self, pred: str, answer: str, question: str) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
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
                
                return {"accuracy":is_correct}, {
                    "match_type": "llm_judge",
                    "judgment": judgment,
                    "explanation": explanation,
                    "qa_judge_model": self.config.qa_judge_model,
                    "qa_raw_response": judge_response
                }
                
            except json.JSONDecodeError as json_error:
                print(f"Failed to parse LLM judge JSON response: {json_error}")
                print(f"Raw response: {judge_response}")
                # Try to extract judgment from raw text as fallback
                judge_response_upper = judge_response.upper()
                if "CORRECT" in judge_response_upper and "INCORRECT" not in judge_response_upper:
                    return {"accuracy": True}, {
                        "match_type": "llm_judge_text_fallback",
                        "judgment": "CORRECT",
                        "explanation": "Extracted from non-JSON response",
                        "qa_judge_model": self.config.qa_judge_model,
                        "qa_raw_response": judge_response
                    }
                else:
                    # Final fallback to string matching
                    return self._fallback_score(pred, answer, perplexity)
            
        except Exception as e:
            print(f"Error in LLM judge scoring: {e}")
            # Fallback to string matching
            return self._fallback_score(pred, answer, perplexity)
    
    def _fallback_score(self, pred: str, answer: str, perplexity: float) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
        """Fallback scoring using string matching when LLM judge fails."""
        
        # Clean up the strings
        pred_clean = pred.strip().lower()
        answer_clean = answer.strip().lower()
        
        # Exact match
        if pred_clean == answer_clean:
            return {"accuracy": True, "perplexity": perplexity}, {"match_type": "exact_fallback", "pred_clean": pred_clean}
        
        # Substring match (answer in prediction)
        if answer_clean in pred_clean:
            return {"accuracy": True, "perplexity": perplexity}, {"match_type": "substring_fallback", "pred_clean": pred_clean}
        
        # Reverse substring match (prediction in answer, for short predictions)
        if len(pred_clean) > 5 and pred_clean in answer_clean:
            return {"accuracy": True, "perplexity": perplexity}, {"match_type": "reverse_substring_fallback", "pred_clean": pred_clean}
        
        # Token overlap similarity (for more flexible matching)
        pred_tokens = set(pred_clean.split())
        answer_tokens = set(answer_clean.split())
        
        if len(answer_tokens) > 0:
            overlap = len(pred_tokens & answer_tokens)
            similarity = overlap / len(answer_tokens)
            
            # Consider it correct if significant overlap (>= 0.6)
            if similarity >= 0.6:
                return {"accuracy": True, "perplexity": perplexity}, {
                    "match_type": "token_overlap_fallback", 
                    "pred_clean": pred_clean,
                    "similarity": similarity,
                }
        
        # No match found
        return {"accuracy": False, "perplexity": perplexity}, {
            "match_type": "no_match_fallback", 
            "pred_clean": pred_clean,
            "similarity": 0.0,
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
        Also calculates perplexity if perplexity judge model is configured.
        """
        
        # Get the question and system prompt for context
        question = ""
        system_prompt = ""
        if hasattr(self, 'qa_items') and self.qa_items:
            # Extract question from convo_id if possible
            try:
                qa_idx = int(convo_id.split('_')[-1]) if '_' in convo_id else 0
                if 0 <= qa_idx < len(self.qa_items):
                    question = self.qa_items[qa_idx]['question']
                    system_prompt = self.qa_items[qa_idx].get('system_prompt', '')
            except (ValueError, IndexError):
                question = "Question context not available"
        
        # Calculate perplexity if perplexity judge is available
        perplexity_scores = {"perplexity": None}
        perplexity_metadata = {}
        if hasattr(self, 'perplexity_judge_client') and self.perplexity_judge_client:
            perplexity_scores, perplexity_metadata = self._perplexity_judge_score(pred, question, system_prompt)
        
        # Use LLM judge if enabled and available
        if self.config.qa_use_llm_judge and hasattr(self, 'qa_judge_client'):
            accuracy_scores, accuracy_metadata = self._llm_judge_score(pred, answer, question)
            # Merge perplexity into the accuracy scores
            accuracy_scores.update(perplexity_scores)
            accuracy_metadata.update(perplexity_metadata)
            return accuracy_scores, accuracy_metadata
        else:
            # Fallback to string matching
            accuracy_scores, accuracy_metadata = self._fallback_score(pred, answer, perplexity_scores.get("perplexity", None))
            accuracy_metadata.update(perplexity_metadata)
            return accuracy_scores, accuracy_metadata