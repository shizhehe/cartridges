"""
Reusable metric calculation functions for evaluation datasets.

Example usage in a dataset class:

```python
from cartridges.metrics import PerplexityMixin
from cartridges.clients.sglang_modal import SGLangClient

class MyDataset(GenerateEvalDataset, PerplexityMixin):
    class Config(GenerateEvalDataset.Config):
        perplexity_judge_model: Optional[SGLangClient.Config] = None
        
    def __init__(self, config, model_helper, seed):
        super().__init__(config, model_helper, seed)
        if self.config.perplexity_judge_model:
            self.perplexity_judge_client = self.config.perplexity_judge_model.instantiate()
            
            # Calculate baselines
            baseline_stats = self._calculate_baseline_perplexities(self.qa_items)
            self.baselines = [baseline_stats] if baseline_stats["perplexity"] else []
    
    def score(self, pred: str, answer: str, convo_id: str):
        # Calculate perplexity alongside other metrics
        perplexity_scores, perplexity_metadata = self._perplexity_judge_score(
            prediction=pred, 
            question=self.get_question(convo_id),
            system_prompt=self.get_system_prompt(convo_id)
        )
        
        # Your other scoring logic here...
        accuracy_scores = {"accuracy": pred == answer}
        
        # Merge scores and metadata
        accuracy_scores.update(perplexity_scores)
        return accuracy_scores, perplexity_metadata
```
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import traceback

from cartridges.clients.sglang_modal import SGLangClient


def calculate_perplexity_sglang(
    client: SGLangClient,
    prediction: str,
    context_messages: List[Dict[str, str]],
    include_tokens: bool = True
) -> Tuple[Dict[str, Optional[float]], Dict[str, Union[str, int, float, List]]]:
    """
    Calculate perplexity of a prediction using SGLang client.
    
    Args:
        client: SGLangClient instance with tokenizer access
        prediction: The text prediction to evaluate
        context_messages: List of messages providing context (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
        include_tokens: Whether to include token-level data in metadata
        
    Returns:
        Tuple of (scores_dict, metadata_dict)
        - scores_dict: {"perplexity": float or None}
        - metadata_dict: Contains method, model, tokens, logprobs, etc.
    """
    try:
        # Construct the full conversation with prediction
        conversation = context_messages + [{"role": "assistant", "content": prediction}]
        
        # Get logprobs from SGLang
        response = client.chat(
            chats=[conversation],
            max_completion_tokens=1,
            temperature=0.0,
            return_logprob=True,
        )
        
        if not response.samples or len(response.samples) == 0:
            return {"perplexity": None}, {"error": "No response samples returned"}
            
        sample = response.samples[0]
        
        # Check for input logprobs
        if not hasattr(sample, 'input_log_prob') or sample.input_log_prob is None:
            return {"perplexity": None}, {"error": "No input logprobs available"}
        
        # Convert to numpy arrays
        if isinstance(sample.input_log_prob, (list, tuple)):
            input_logprobs = np.array(sample.input_log_prob)
            tokens = sample.tokens
        else:
            input_logprobs = sample.input_log_prob
            tokens = sample.tokens
        
        # Get context tokens to identify prediction portion
        context_tokens = client.tokenizer.apply_chat_template(
            context_messages,
            add_generation_prompt=True,
            return_tensors=None,
            tokenize=True
        )
        num_context_tokens = len(context_tokens)
        
        # Extract prediction tokens and logprobs
        if len(input_logprobs) <= num_context_tokens:
            return {"perplexity": None}, {"error": "No prediction tokens found in logprobs"}
        
        prediction_logprobs = input_logprobs[num_context_tokens:]
        prediction_tokens = tokens[num_context_tokens:]
        
        # Align with actual prediction length
        prediction_token_length = len(client.tokenizer.tokenize(prediction))
        prediction_logprobs = prediction_logprobs[:prediction_token_length]
        prediction_tokens = prediction_tokens[:prediction_token_length]
        
        # Filter valid logprobs
        valid_logprobs = [lp for lp in prediction_logprobs if not np.isnan(lp) and not np.isinf(lp)]
        
        if len(valid_logprobs) == 0:
            return {"perplexity": None}, {"error": "No valid logprobs found"}
        
        # Calculate perplexity
        mean_logprob = np.mean(valid_logprobs)
        perplexity = np.exp(-mean_logprob)
        
        # Build metadata (ensure all values are JSON serializable)
        metadata = {
            "method": "sglang_prediction_logprobs",
            "perplexity_judge_model": str(client.model_name),
            "num_prediction_tokens": int(len(valid_logprobs)),
            "mean_logprob": float(mean_logprob),
        }
        
        # Optionally include token-level data
        if include_tokens:
            metadata.update({
                "prediction_tokens": [str(token) for token in prediction_tokens],
                "prediction_logprobs": [float(lp) for lp in prediction_logprobs],
            })
        
        return {"perplexity": float(perplexity)}, metadata
        
    except Exception as e:
        return {"perplexity": None}, {"perplexity_error": f"Perplexity calculation failed: {str(e)}"}


def calculate_baseline_perplexities(
    client: SGLangClient,
    qa_items: List[Dict[str, str]],
    context_key: str = "system_prompt",
    question_key: str = "question", 
    answer_key: str = "answer"
) -> Dict[str, float]:
    """
    Calculate baseline perplexity statistics for a dataset.
    
    Args:
        client: SGLangClient instance
        qa_items: List of QA items, each containing context, question, and answer
        context_key: Key for context/system prompt in each item
        question_key: Key for question in each item  
        answer_key: Key for ground truth answer in each item
        
    Returns:
        Dictionary with baseline statistics (mean, std, etc.)
    """
    baseline_scores = []
    
    for item in qa_items:
        context_messages = []
        if context_key in item and item[context_key]:
            context_messages.append({"role": "system", "content": item[context_key]})
        context_messages.append({"role": "user", "content": item[question_key]})
        
        scores, _ = calculate_perplexity_sglang(
            client=client,
            prediction=item[answer_key],
            context_messages=context_messages,
            include_tokens=False  # Don't need tokens for baseline calculation
        )
        
        if scores["perplexity"] is not None:
            baseline_scores.append(scores["perplexity"])
    
    if not baseline_scores:
        return {"perplexity": None, "count": 0}
    
    return {
        "perplexity": float(np.mean(baseline_scores)),
        "perplexity_std": float(np.std(baseline_scores)),
        "perplexity_min": float(np.min(baseline_scores)),
        "perplexity_max": float(np.max(baseline_scores)),
        "count": len(baseline_scores),
        "total_items": len(qa_items)
    }


class PerplexityMixin:
    """
    Mixin class that datasets can inherit to easily add perplexity scoring.
    
    Requires the dataset to have:
    - self.config.perplexity_judge_model (SGLangClient.Config)
    - self.perplexity_judge_client (instantiated SGLangClient)
    """
    
    def _perplexity_judge_score(
        self, 
        prediction: str, 
        question: str, 
        system_prompt: str = ""
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Union[str, int, float, List]]]:
        """
        Calculate perplexity score for a prediction given context.
        
        Args:
            prediction: Model's prediction text
            question: Question/prompt text
            system_prompt: System prompt (optional)
            
        Returns:
            Tuple of (scores_dict, metadata_dict)
        """
        if not hasattr(self, 'perplexity_judge_client') or not self.perplexity_judge_client:
            return {"perplexity": None}, {"error": "No perplexity judge client configured"}
        
        context_messages = []
        if system_prompt:
            context_messages.append({"role": "system", "content": system_prompt})
        context_messages.append({"role": "user", "content": question})
        
        return calculate_perplexity_sglang(
            client=self.perplexity_judge_client,
            prediction=prediction,
            context_messages=context_messages,
            include_tokens=True
        )
    
    def _calculate_baseline_perplexities(self, qa_items: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Calculate baseline perplexity statistics for the dataset.
        
        Args:
            qa_items: List of QA items with 'question', 'answer', 'system_prompt' keys
            
        Returns:
            Dictionary with baseline statistics
        """
        if not hasattr(self, 'perplexity_judge_client') or not self.perplexity_judge_client:
            return {"perplexity": None, "count": 0}
        
        return calculate_baseline_perplexities(
            client=self.perplexity_judge_client,
            qa_items=qa_items,
            context_key="system_prompt",
            question_key="question",
            answer_key="answer"
        )