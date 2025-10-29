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

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert at creating question-answer pairs about email data for evaluation purposes.

Below is a selection of emails from {name}'s mailbox (User ID: {user_id}).
This user has {total_emails} emails in this batch.

{emails}

Your task is to create high-quality question-answer pairs that test understanding of this email data. Focus on:
1. Factual questions about email content, senders, recipients, dates, subjects
2. Questions about relationships and communication patterns
3. Questions about specific details mentioned in the emails
4. Questions that require understanding context across multiple emails

Generate questions that can ONLY be answered from the provided email context above."""

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

class EnronStreamQAResource(Resource):
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
        
        # QA-specific seed prompts
        seed_prompts: List[SEED_TYPES] = [
            "question",        # Direct factual questions
            "analytical",      # Analysis questions  
            "factual",         # Factual information extraction
            "summarization",   # Summary questions
        ]
        
        # Multi-batch synthesis options
        synthesize_all_batches: bool = True  # If True, creates separate QA datasets for each batch
        
        # Target batch ID for single batch generation
        target_batch_id: Optional[int] = None  # If set, only generate for this specific batch
        
    def __init__(self, config: Config):
        self.config = config
        self._current_batch_id = config.target_batch_id if config.target_batch_id is not None else 0
        
        # Load batch contexts if path provided, otherwise create them
        if self.config.batch_contexts_path and Path(self.config.batch_contexts_path).exists():
            self._load_existing_contexts()
        else:
            self._create_temporal_batches()
        
        logger.info(f"EnronStreamQAResource initialized for user {self.config.user_id}")
        logger.info(f"Total batches available: {len(self.batch_contexts)}")
        if hasattr(self, '_current_batch_id'):
            logger.info(f"Current batch: {self._current_batch_id}")
            if self._current_batch_id < len(self.batch_contexts):
                current_chars = len(self.batch_contexts[self._current_batch_id]['batch_content'])
                logger.info(f"Current batch has {current_chars} chars")
    
    def _load_existing_contexts(self):
        """Load existing batch contexts from JSON file."""
        logger.info(f"Loading existing contexts from {self.config.batch_contexts_path}")
        with open(self.config.batch_contexts_path, 'r', encoding='utf-8') as f:
            self.batch_contexts = json.load(f)
        logger.info(f"Loaded {len(self.batch_contexts)} batch contexts")
    
    def _create_temporal_batches(self):
        """Create temporal batches from Enron data."""
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
        batch_header = f"EMAIL BATCH {batch_metadata.get('batch_id', 'UNKNOWN')}\\n"
        batch_header += f"Time Period: {batch_metadata['start_date']} to {batch_metadata['end_date']}\\n"
        batch_header += f"Total Emails: {len(sorted_emails)}\\n"
        batch_header += f"Users: {', '.join(sorted(user_names))}\\n"
        batch_header += "=" * 80 + "\\n\\n"
        
        # Format all emails in the batch
        formatted_emails = []
        for i, (email_path, email) in enumerate(sorted_emails, 1):
            truncated_body = self._truncate_email(email.body)
            
            formatted_email = f"EMAIL {i}/{len(sorted_emails)}\\n"
            formatted_email += EMAIL_TEMPLATE.format(
                email_id=email.email_id,
                from_addr=email.from_addr,
                to_addr=email.to_addr,
                subject=email.subject,
                date=email.date,
                folder=self.folder_name_mapping.get(email.folder, email.folder),
                body=truncated_body
            )
            formatted_email += "\\n" + "-" * 60 + "\\n"
            
            formatted_emails.append(formatted_email)
        
        # Combine everything
        return batch_header + "\\n".join(formatted_emails)
    
    def _truncate_email(self, email_body: str) -> str:
        """Truncate email body if too long."""
        if self.config.max_chars_per_email is None or len(email_body) <= self.config.max_chars_per_email:
            return email_body

        start_idx = random.randint(0, len(email_body) - self.config.max_chars_per_email)
        end_idx = start_idx + self.config.max_chars_per_email
        truncated = email_body[start_idx:end_idx]
        return f"... {truncated} ..."
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str], List[str]]:
        """Sample prompts for QA generation."""
        # Get current batch content
        current_context = self.batch_contexts[self._current_batch_id]['batch_content']
        
        # Sample content from current context if chunker is available
        if self.config.chunker:
            chunker = self.config.chunker.instantiate(text=current_context)
            sampled_current_context = chunker.sample_chunk()
        else:
            sampled_current_context = current_context
        
        # Get user information for the system prompt
        total_emails = self.batch_contexts[self._current_batch_id].get('num_emails', 0)
        
        # Format with QA-specific system prompt template
        qa_content = SYSTEM_PROMPT_TEMPLATE.format(
            name=self.config.user_id,  # Use user_id as name since we don't have full name
            user_id=self.config.user_id,
            total_emails=total_emails,
            emails=sampled_current_context
        )
        
        # Sample seed prompts for QA generation
        seed_prompts, seed_prompt_types = sample_seed_prompts(self.config.seed_prompts, batch_size)
        
        return qa_content, seed_prompts, seed_prompt_types
    
    async def setup(self):
        """Setup method called by synthesis framework."""
        pass
    
    def get_synthesis_variants(self) -> List['EnronStreamQAResource']:
        """Return a list of resource variants for multi-batch QA synthesis."""
        if not self.config.synthesize_all_batches:
            return [self]
        
        variants = []
        for batch_id in range(len(self.batch_contexts)):
            # Create a copy of the resource for this batch
            batch_resource = EnronStreamQAResource(self.config)
            batch_resource.set_batch_id(batch_id)
            # Add batch identifier for output naming
            batch_resource._batch_suffix = f"_batch_{batch_id}"
            variants.append(batch_resource)
        
        logger.info(f"Created {len(variants)} QA resource variants for batches 0-{len(variants)-1}")
        return variants
    
    def get_output_suffix(self) -> str:
        """Get suffix for output naming."""
        return getattr(self, '_batch_suffix', '')
    
    def set_batch_id(self, batch_id: int):
        """Set which batch to use for QA generation."""
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