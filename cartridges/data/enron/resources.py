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
Below is a selection of emails from {name}'s mailbox (User ID: {user_id}).
This user has {total_emails} emails across {num_folders} folders.
{emails}"""

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

FULL_STRING_TEMPLATE = """\
<user-mailbox-{user_id}>
Below is the complete email history for {name} (User ID: {user_id}).
This user has {total_emails} emails across {num_folders} folders.
<emails>
{emails}
</emails>
</user-mailbox-{user_id}>"""

class EnronStreamResource(Resource):
    class Config(Resource.Config):
        user_id: str  # Single user ID to process
        max_emails_per_batch: int = 1000
        max_emails_per_prompt: int = 5
        min_emails_per_prompt: int = 1
        max_chars_per_email: Optional[int] = 1000
        folders_to_include: Optional[List[str]] = None  # e.g., ["inbox", "sent"]
        seed_prompts: List[SEED_TYPES] = ["generic"]
        dataset_path: str = "datasets/enron"
        selected_users_file: Optional[str] = None  # Path to selected_users.json
        num_batches: Optional[int] = None  # Total number of batches to create (None = use all data)
        metadata_output_path: Optional[str] = None  # Where to save batch metadata
        
        # Chunker configuration for temporal context sampling
        chunker: Optional[Chunker.Config] = None
        
        # Context saving options
        save_full_context: bool = True  # Save complete context across all batches
        save_batch_contexts: bool = True  # Save individual batch contexts
        context_output_dir: Optional[str] = None  # Where to save context files
        synthesis_output_dir: Optional[str] = None  # Synthesis config output_dir
        
        # New config option to enable multi-batch synthesis
        synthesize_all_batches: bool = True  # If True, creates separate datasets for each batch
        batch_output_dir: Optional[str] = None  # Base directory for batch outputs
        
    def __init__(self, config: Config):
        self.config = config
        self.chunkers = {}  # Store chunkers for each batch

        # Check if a specific batch ID was set for this instance
        if hasattr(config, '_target_batch_id'):
            self._current_batch_id = config._target_batch_id

        # Load folder name mappings if selected_users_file is provided
        self.folder_name_mapping = {}
        if self.config.selected_users_file:
            try:
                with open(self.config.selected_users_file, 'r') as f:
                    selected_users_data = json.load(f)
                    if self.config.user_id in selected_users_data:
                        self.folder_name_mapping = selected_users_data[self.config.user_id]
                        logger.info(f"Loaded folder name mappings for {self.config.user_id}: {self.folder_name_mapping}")
            except Exception as e:
                logger.warning(f"Could not load folder name mappings: {e}")

        # Load single user based on config
        if self.config.selected_users_file:
            # Load from selected users file, but filter to single user
            all_users = load_enron_selected_users(
                selected_users_path=self.config.selected_users_file,
                dataset_path=self.config.dataset_path
            )
            # Filter to the specified user
            self.users = [user for user in all_users if user.user_id == self.config.user_id]
            if not self.users:
                raise ValueError(f"User {self.config.user_id} not found in selected users file")
        else:
            # Load using explicit user_id and folders
            self.users = load_enron_user_data(
                user_ids=[self.config.user_id],
                dataset_path=self.config.dataset_path,
                folders_to_include=self.config.folders_to_include
            )
        
        # Create temporal batches
        self.batches, self.batch_metadata = create_temporal_batches(
            users=self.users,
            max_emails_per_batch=self.config.max_emails_per_batch,
            num_batches=self.config.num_batches
        )
        
        # Save metadata if path specified
        if self.config.metadata_output_path:
            self._save_batch_metadata()
        
        # Initialize chunkers for each batch if chunker config is provided
        if self.config.chunker:
            self._initialize_chunkers()
        
        # Save contexts if requested
        if self.config.save_full_context or self.config.save_batch_contexts:
            self._save_contexts()
        
        logger.info(f"Created {len(self.batches)} temporal batches for user {self.config.user_id}")
        for i, metadata in enumerate(self.batch_metadata):
            logger.info(f"Batch {i}: {metadata['num_emails']} emails from {metadata['start_date']} to {metadata['end_date']}")
    
    def _save_batch_metadata(self):
        """Save batch metadata to file."""
        metadata_path = Path(self.config.metadata_output_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.batch_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved batch metadata to {metadata_path}")
    
    def _initialize_chunkers(self):
        """Initialize chunkers for each batch."""
        logger.info("Initializing chunkers for temporal batches...")
        for batch_id in range(len(self.batches)):
            batch_content = self._get_batch_content(batch_id)
            chunker = self.config.chunker.instantiate(text=batch_content)
            self.chunkers[batch_id] = chunker
            logger.info(f"Initialized chunker for batch {batch_id} with {len(batch_content)} characters")
    
    def _save_contexts(self):
        """Save full context and individual batch contexts to JSON files."""
        # Determine output directory
        if self.config.context_output_dir:
            context_dir = Path(self.config.context_output_dir)
        elif self.config.synthesis_output_dir:
            # Use synthesis config output_dir
            context_dir = Path(self.config.synthesis_output_dir)
        else:
            # Fallback to environment variable
            output_base = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")
            context_dir = Path(output_base) / "enron" / self.config.user_id
        
        context_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full context across all batches
        if self.config.save_full_context:
            logger.info("Saving full context across all batches...")
            full_context = self._get_all_batches_content()
            
            full_context_data = {
                "user_id": self.config.user_id,
                "total_batches": len(self.batches),
                "total_emails": sum(len(batch) for batch in self.batches),
                "time_range": {
                    "start": str(min(metadata['start_date'] for metadata in self.batch_metadata)),
                    "end": str(max(metadata['end_date'] for metadata in self.batch_metadata))
                },
                "chunker_config": self.config.chunker.model_dump() if self.config.chunker else None,
                "full_content": full_context,
                "content_length_chars": len(full_context),
                "content_length_tokens": len(self.chunkers[0].tokens) if self.chunkers else None
            }
            
            full_context_path = context_dir / f"full_context-{self.config.user_id}.json"
            with open(full_context_path, 'w', encoding='utf-8') as f:
                json.dump(full_context_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved full context to {full_context_path}")
        
        # Save individual batch contexts
        if self.config.save_batch_contexts:
            logger.info("Saving individual batch contexts...")
            batch_contexts = []
            
            for batch_id in range(len(self.batches)):
                batch_content = self._get_batch_content(batch_id)
                batch_info = self.get_batch_info(batch_id)
                
                batch_context_data = {
                    "batch_id": batch_id,
                    "user_id": self.config.user_id,
                    "num_emails": batch_info['num_emails'],
                    "time_range": {
                        "start": str(batch_info['start_date']),
                        "end": str(batch_info['end_date'])
                    },
                    "users": batch_info['users'],
                    "email_paths": batch_info['email_paths'],
                    "batch_content": batch_content,
                    "content_length_chars": len(batch_content),
                    "content_length_tokens": len(self.chunkers[batch_id].tokens) if batch_id in self.chunkers else None
                }
                
                batch_contexts.append(batch_context_data)
            
            batch_contexts_path = context_dir / f"batch_contexts-{self.config.user_id}.json"
            with open(batch_contexts_path, 'w', encoding='utf-8') as f:
                json.dump(batch_contexts, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved {len(batch_contexts)} batch contexts to {batch_contexts_path}")
    
    def get_batch_info(self, batch_id: int) -> Dict[str, Any]:
        """Get information about a specific batch."""
        if 0 <= batch_id < len(self.batch_metadata):
            return self.batch_metadata[batch_id]
        else:
            raise ValueError(f"Batch {batch_id} does not exist")
    
    def get_batch_emails(self, batch_id: int) -> List[Tuple[str, Any]]:
        """Get all emails in a specific batch."""
        if batch_id >= len(self.batches):
            raise ValueError(f"Batch {batch_id} does not exist")
        
        return self.batches[batch_id]
    
    def _truncate_email(self, email_body: str) -> str:
        """Truncate email body if too long."""
        if self.config.max_chars_per_email is None or len(email_body) <= self.config.max_chars_per_email:
            return email_body

        start_idx = random.randint(0, len(email_body) - self.config.max_chars_per_email)
        end_idx = start_idx + self.config.max_chars_per_email
        truncated = email_body[start_idx:end_idx]
        return f"... {truncated} ..."

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        # Get content for the specified batch (set via set_batch_id)
        batch_id = getattr(self, '_current_batch_id', 0)
        
        # Use chunker if available, otherwise fall back to full batch content
        if self.config.chunker and batch_id in self.chunkers:
            content = self.chunkers[batch_id].sample_chunk()
        else:
            content = self._get_batch_content(batch_id)
        
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return content, seed_prompts
    
    def set_batch_id(self, batch_id: int):
        """Set which batch to use for synthesis."""
        if 0 <= batch_id < len(self.batches):
            self._current_batch_id = batch_id
            logger.info(f"Set current batch to {batch_id}")
        else:
            raise ValueError(f"Batch {batch_id} does not exist. Available batches: 0-{len(self.batches)-1}")
    
    def get_num_batches(self) -> int:
        """Get the total number of batches."""
        return len(self.batches)
    
    def _get_human_readable_folder(self, folder_name: str) -> str:
        """Get human-readable folder name from mapping, fallback to original."""
        return self.folder_name_mapping.get(folder_name, folder_name)
    
    def get_synthesis_variants(self) -> List['EnronStreamResource']:
        """Return a list of resource variants for multi-batch synthesis."""
        if not self.config.synthesize_all_batches:
            return [self]
        
        variants = []
        for batch_id in range(len(self.batches)):
            # Create a copy of the resource for this batch
            batch_resource = EnronStreamResource(self.config)
            batch_resource.set_batch_id(batch_id)
            # Add batch identifier for output naming
            batch_resource._batch_suffix = f"_batch_{batch_id}"
            variants.append(batch_resource)
        
        return variants
    
    def get_output_suffix(self) -> str:
        """Get suffix for output naming."""
        return getattr(self, '_batch_suffix', '')
    
    async def setup(self):
        """Async setup method for initializing chunkers if needed."""
        # This method is called by the synthesis framework
        # Chunkers are already initialized in __init__, but we provide this for consistency
        if self.config.chunker and not self.chunkers:
            self._initialize_chunkers()
    
    
    def _get_batch_content(self, batch_id: int) -> str:
        """Get all emails in a specific batch as a single formatted context."""
        batch_emails = self.get_batch_emails(batch_id)
        
        if not batch_emails:
            raise ValueError(f"No emails in batch {batch_id}")
        
        # Sort emails by date within the batch for chronological order
        sorted_emails = sorted(batch_emails, 
                             key=lambda x: x[1].parsed_date if x[1].parsed_date else datetime.min)
        
        batch_info = self.get_batch_info(batch_id)
        user_names = set(email.user_id for _, email in sorted_emails)
        
        # Create batch header
        batch_header = f"EMAIL BATCH {batch_id}\n"
        batch_header += f"Time Period: {batch_info['start_date']} to {batch_info['end_date']}\n"
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
                folder=self._get_human_readable_folder(email.folder),
                body=truncated_body
            )
            formatted_email += "\n" + "-" * 60 + "\n"
            
            formatted_emails.append(formatted_email)
        
        # Combine everything
        full_context = batch_header + "\n".join(formatted_emails)
        
        return full_context
    
    def _get_all_batches_content(self) -> str:
        """Get all emails from all batches as a single formatted context."""
        if not self.batches:
            raise ValueError("No batches available")
        
        all_batch_contents = []
        
        # Process each batch
        for batch_id in range(len(self.batches)):
            batch_content = self._get_batch_content(batch_id)
            all_batch_contents.append(batch_content)
        
        # Create overall header
        total_emails = sum(len(batch) for batch in self.batches)
        earliest_date = min(metadata['start_date'] for metadata in self.batch_metadata)
        latest_date = max(metadata['end_date'] for metadata in self.batch_metadata)
        
        overall_header = f"ENRON EMAIL DATASET - USER {self.config.user_id.upper()} - ALL BATCHES\n"
        overall_header += f"User: {self.config.user_id}\n"
        overall_header += f"Total Batches: {len(self.batches)}\n"
        overall_header += f"Time Period: {earliest_date} to {latest_date}\n"
        overall_header += f"Total Emails: {total_emails}\n"
        overall_header += "=" * 100 + "\n\n"
        
        # Combine all batches
        return overall_header + "\n\n".join(all_batch_contents)

    def to_string(self) -> str:
        out = "Below is a collection of Enron employee email histories."
        
        for user in self.users:
            formatted_emails = []
            for email_id, email in user.emails.items():
                formatted_email = f"<{email_id}>\nFrom: {email.from_addr}\nTo: {email.to_addr}\nSubject: {email.subject}\nDate: {email.date}\nFolder: {self._get_human_readable_folder(email.folder)}\n\n{email.body}\n</{email_id}>"
                formatted_emails.append(formatted_email)
            
            emails_text = "\n\n".join(formatted_emails)
            
            out += "\n\n"
            out += FULL_STRING_TEMPLATE.format(
                name=user.name,
                user_id=user.user_id,
                total_emails=len(user.emails),
                num_folders=len(user.folders),
                emails=emails_text
            )
        
        return out