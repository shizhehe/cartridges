from typing import Dict, List, Optional, Tuple, Any
import random
import json
from pathlib import Path
from datetime import datetime

from cartridges.utils import get_logger
from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES, Chunker
from cartridges.data.chunkers import TokenChunker
from cartridges.data.enron.utils import load_enron_user_data, load_enron_selected_users, create_temporal_batches
import os

logger = get_logger(__name__)


EMAIL_TEMPLATE = """\
<email-{email_id}>
From: {from_addr}
To: {to_addr}
Subject: {subject}
Date: {date}
Folder: {folder}

{body}
</email-{email_id}>
"""

class EnronStreamEvalResource(Resource):
    class Config(Resource.Config):
        user_id: str  # Single user ID to process
        max_emails_per_batch: int = 1000
        max_chars_per_email: Optional[int] = 1000
        folders_to_include: Optional[List[str]] = None  # e.g., ["inbox", "sent"]
        dataset_path: str = "datasets/enron"
        selected_users_file: Optional[str] = None  # Path to selected_users.json
        num_batches: Optional[int] = None  # Total number of batches to create (None = use all data)
        
        
        # Chunker configuration for sampling from current batch
        chunker: Optional[Chunker.Config] = None
        
        # Context paths - can override if contexts already exist
        batch_contexts_path: Optional[str] = None  # Path to existing batch_contexts-{user_id}.json
        
        # Seed prompts for evaluation (different from training)
        seed_prompts: List[SEED_TYPES] = ["question", "analytical", "factual"]
        
        # Multi-batch synthesis options (similar to EnronStreamResource)
        synthesize_all_batches: bool = True  # If True, creates separate eval datasets for each batch
        
    def __init__(self, config: Config):
        self.config = config
        self._current_batch_id = 0  # Default to first batch
        
        # Load batch contexts if path provided, otherwise create them
        if self.config.batch_contexts_path and Path(self.config.batch_contexts_path).exists():
            self._load_existing_contexts()
        else:
            self._create_temporal_batches()
        
        logger.info(f"EnronStreamEvalResource initialized for user {self.config.user_id}")
        logger.info(f"Total batches available: {len(self.batch_contexts)}")
        if hasattr(self, '_current_batch_id'):
            logger.info(f"Current batch: {self._current_batch_id}")
            if self._current_batch_id < len(self.batch_contexts):
                current_chars = len(self.batch_contexts[self._current_batch_id]['batch_content'])
                logger.info(f"Current batch has {current_chars} chars")
                if self._current_batch_id < len(self.batch_contexts) - 1:
                    future_chars = sum(len(ctx['batch_content']) for ctx in self.batch_contexts[self._current_batch_id + 1:])
                    logger.info(f"Future context has {future_chars} chars")
    
    def _load_existing_contexts(self):
        """Load existing batch contexts from JSON file."""
        logger.info(f"Loading existing contexts from {self.config.batch_contexts_path}")
        with open(self.config.batch_contexts_path, 'r', encoding='utf-8') as f:
            self.batch_contexts = json.load(f)
        logger.info(f"Loaded {len(self.batch_contexts)} batch contexts")
    
    def _create_temporal_batches(self):
        """Create temporal batches from Enron data (similar to EnronStreamResource)."""
        logger.info("Creating temporal batches from Enron data...")
        
        # Load folder name mappings if selected_users_file is provided
        self.folder_name_mapping = {}
        if self.config.selected_users_file:
            try:
                with open(self.config.selected_users_file, 'r') as f:
                    selected_users_data = json.load(f)
                    if self.config.user_id in selected_users_data:
                        self.folder_name_mapping = selected_users_data[self.config.user_id]
            except Exception as e:
                logger.warning(f"Could not load folder name mappings: {e}")

        # Load single user based on config
        if self.config.selected_users_file:
            all_users = load_enron_selected_users(
                selected_users_path=self.config.selected_users_file,
                dataset_path=self.config.dataset_path
            )
            users = [user for user in all_users if user.user_id == self.config.user_id]
            if not users:
                raise ValueError(f"User {self.config.user_id} not found in selected users file")
        else:
            users = load_enron_user_data(
                user_ids=[self.config.user_id],
                dataset_path=self.config.dataset_path,
                folders_to_include=self.config.folders_to_include
            )
        
        # Create temporal batches
        batches, batch_metadata = create_temporal_batches(
            users=users,
            max_emails_per_batch=self.config.max_emails_per_batch,
            num_batches=self.config.num_batches
        )
        
        # Convert to batch contexts format
        self.batch_contexts = []
        for batch_id in range(len(batches)):
            batch_content = self._format_batch_content(batches[batch_id], batch_metadata[batch_id])
            
            batch_context_data = {
                "batch_id": batch_id,
                "user_id": self.config.user_id,
                "num_emails": batch_metadata[batch_id]['num_emails'],
                "time_range": {
                    "start": str(batch_metadata[batch_id]['start_date']),
                    "end": str(batch_metadata[batch_id]['end_date'])
                },
                "batch_content": batch_content,
                "content_length_chars": len(batch_content),
            }
            self.batch_contexts.append(batch_context_data)
        
        logger.info(f"Created {len(self.batch_contexts)} batch contexts")
    
    def _format_batch_content(self, batch_emails, batch_metadata):
        """Format batch emails into content string."""
        if not batch_emails:
            return ""
        
        # Sort emails by date within the batch for chronological order
        sorted_emails = sorted(batch_emails, 
                             key=lambda x: x[1].parsed_date if x[1].parsed_date else datetime.min)
        
        user_names = set(email.user_id for _, email in sorted_emails)
        
        # Create batch header
        batch_header = f"EMAIL BATCH {batch_metadata.get('batch_id', 'UNKNOWN')}\n"
        batch_header += f"Time Period: {batch_metadata['start_date']} to {batch_metadata['end_date']}\n"
        batch_header += f"Total Emails: {len(sorted_emails)}\n"
        batch_header += f"Users: {', '.join(sorted(user_names))}\n"
        batch_header += "=" * 80 + "\n\n"
        
        # Format all emails in the batch
        formatted_emails = []
        for i, (email_path, email) in enumerate(sorted_emails, 1):
            truncated_body = self._truncate_email(email.body)
            
            formatted_email = f"EMAIL {i}/{len(sorted_emails)}\n"
            formatted_email += EMAIL_TEMPLATE.format(
                email_id=email.email_id,
                from_addr=email.from_addr,
                to_addr=email.to_addr,
                subject=email.subject,
                date=email.date,
                folder=self.folder_name_mapping.get(email.folder, email.folder),
                body=truncated_body
            )
            formatted_email += "\n" + "-" * 60 + "\n"
            
            formatted_emails.append(formatted_email)
        
        # Combine everything
        return batch_header + "\n".join(formatted_emails)
    
    def _truncate_email(self, email_body: str) -> str:
        """Truncate email body if too long."""
        if self.config.max_chars_per_email is None or len(email_body) <= self.config.max_chars_per_email:
            return email_body

        start_idx = random.randint(0, len(email_body) - self.config.max_chars_per_email)
        end_idx = start_idx + self.config.max_chars_per_email
        truncated = email_body[start_idx:end_idx]
        return f"... {truncated} ..."
    
    
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        """Sample prompts with temporal awareness for evaluation dataset generation."""
        # Get current batch content
        current_context = self.batch_contexts[self._current_batch_id]['batch_content']
        
        # Get future context (all batches after current batch) as distractor
        future_batches = self.batch_contexts[self._current_batch_id + 1:]
        if future_batches:
            future_context = "\n\n".join([
                f"Future Context {j+1}:\n{batch_dict['batch_content']}" 
                for j, batch_dict in enumerate(future_batches)
            ])
        else:
            future_context = "No future context available."
        
        # Sample content from current context if chunker is available
        if self.config.chunker:
            chunker = self.config.chunker.instantiate(text=current_context)
            sampled_current_context = chunker.sample_chunk()
        else:
            sampled_current_context = current_context
        
        # Create temporally-aware prompt template for synthesis framework
        # This mimics your original QA_GENERATION_PROMPT structure but works with seed prompts
        temporal_content = f"""## Instructions for Conversation Generation:
You are generating conversations about email data with temporal awareness. You have access to two contexts:

### Context A (Distractor - Future Information):
{future_context}

### Context B (Relevant - Current Information):
{sampled_current_context}

### Critical Requirements:
1. Generate conversations that can ONLY be answered or discussed using information from Context B (Current Information)
2. DO NOT use any information from Context A (Future Information) in your responses
3. Context A represents information that should not be available at this time period
4. Focus on understanding, analyzing, and discussing the emails in Context B
5. Ensure that any questions, analysis, or discussion points can be resolved using only Context B

### Temporal Context:
This is batch {self._current_batch_id} of the email timeline. When generating conversations, maintain awareness that this represents a specific time period and future information should not influence the discussion.
"""
        
        # Sample seed prompts - these will determine the TYPE of conversations generated
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        
        return temporal_content, seed_prompts
    
    async def setup(self):
        """Setup method called by synthesis framework."""
        # No need to generate QA pairs - synthesis framework will handle it
        pass
    
    def get_synthesis_variants(self) -> List['EnronStreamEvalResource']:
        """Return a list of resource variants for multi-batch eval synthesis."""
        if not self.config.synthesize_all_batches:
            return [self]
        
        variants = []
        for batch_id in range(len(self.batch_contexts)):
            # Create a copy of the resource for this batch (same pattern as EnronStreamResource)
            batch_resource = EnronStreamEvalResource(self.config)
            batch_resource.set_batch_id(batch_id)
            # Add batch identifier for output naming
            batch_resource._batch_suffix = f"_batch_{batch_id}"
            variants.append(batch_resource)
        
        logger.info(f"Created {len(variants)} eval resource variants for batches 0-{len(variants)-1}")
        return variants
    
    def get_output_suffix(self) -> str:
        """Get suffix for output naming."""
        return getattr(self, '_batch_suffix', '')
    
    def set_batch_id(self, batch_id: int):
        """Set which batch to use for eval generation."""
        if 0 <= batch_id < len(self.batch_contexts):
            self._current_batch_id = batch_id
            logger.info(f"Set current batch to {batch_id}")
        else:
            raise ValueError(f"Batch {batch_id} does not exist. Available batches: 0-{len(self.batch_contexts)-1}")
    
    def get_num_batches(self) -> int:
        """Get the total number of batches."""
        return len(self.batch_contexts)
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about the current batch."""
        return self.batch_contexts[self._current_batch_id]
    
