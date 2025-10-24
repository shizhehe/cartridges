import os
import random
import sqlite3
import asyncio
import math
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from cartridges.data.resources import Resource
from cartridges.utils import get_logger

logger = get_logger(__name__)


@dataclass
class iMessageThread:
    """Represents an iMessage conversation thread."""
    chat_identifier: str
    display_name: str
    messages: List[Dict[str, Any]]
    participant_count: int
    last_message_date: datetime
    total_messages: int


class iMessageResource(Resource):
    
    class Config(Resource.Config):
        # Path to iMessage database (default macOS location)
        db_path: str = os.path.expanduser("~/Library/Messages/chat.db")
        
        # Date filtering
        date_start: Optional[str] = None  # Format: YYYY-MM-DD
        date_end: Optional[str] = None
        date_days_in_bucket: int = 30
        
        # Conversation filtering
        min_messages_per_thread: int = 5
        max_messages_per_thread: int = 100
        include_group_chats: bool = True
        include_individual_chats: bool = True
        
        # Contact filtering (optional list of phone numbers or handles)
        include_contacts: Optional[List[str]] = None
        exclude_contacts: Optional[List[str]] = None
        
        # Temporal bias for sampling recent conversations
        temporal_decay_rate: float = 0.1
        temporal_half_life_days: Optional[int] = None
        
        # Cache directory for processed threads
        cache_dir: str = os.path.expanduser("~/.cartridges/data/imessage")

    def __init__(self, config: Config):
        self.config = config
        self.threads: List[iMessageThread] = []
        self.threads_by_bucket: Dict[str, List[iMessageThread]] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)

    async def setup(self):
        """Load and process iMessage threads from the database."""
        if not os.path.exists(self.config.db_path):
            raise FileNotFoundError(f"iMessage database not found at {self.config.db_path}")
        
        logger.info(f"Loading iMessage threads from {self.config.db_path}")
        
        # Load threads from database
        self.threads = await self._load_threads_from_db()
        
        # Organize threads by date buckets
        self._organize_threads_by_bucket()
        
        logger.info(f"Loaded {len(self.threads)} iMessage threads across {len(self.threads_by_bucket)} date buckets")

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        """Sample an iMessage thread and return prompts for synthesis."""
        if not self.threads:
            raise ValueError("No threads loaded. Call setup() first.")
        
        # Sample a thread with temporal bias
        sampled_thread = await self._sample_thread()
        
        # Format thread content for synthesis
        thread_content = self._format_thread_content(sampled_thread)
        
        # Sample prompts from predefined list
        prompts = random.choices(IMESSAGE_PROMPTS, k=batch_size)
        
        return thread_content, prompts

    async def _load_threads_from_db(self) -> List[iMessageThread]:
        """Load iMessage threads from SQLite database."""
        threads = []
        
        try:
            # Connect to iMessage database (read-only)
            conn = sqlite3.connect(f"file:{self.config.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Query to get chat information with message counts
            chat_query = """
            SELECT 
                c.chat_identifier,
                c.display_name,
                c.service_name,
                COUNT(m.ROWID) as message_count,
                MAX(m.date) as last_message_date,
                c.is_archived,
                c.is_filtered
            FROM chat c
            LEFT JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
            LEFT JOIN message m ON cmj.message_id = m.ROWID
            WHERE c.service_name IN ('iMessage', 'SMS')
            GROUP BY c.ROWID
            HAVING message_count >= ?
            ORDER BY last_message_date DESC
            """
            
            cursor.execute(chat_query, (self.config.min_messages_per_thread,))
            chat_results = cursor.fetchall()
            
            for chat_data in chat_results:
                chat_identifier, display_name, service_name, message_count, last_message_date, is_archived, is_filtered = chat_data
                
                # Skip if filtering criteria not met
                if not self._should_include_chat(chat_identifier, message_count):
                    continue
                
                # Load messages for this chat
                messages = await self._load_messages_for_chat(cursor, chat_identifier)
                
                if messages:
                    # Convert Core Data timestamp to datetime
                    last_date = self._convert_core_data_timestamp(last_message_date)
                    
                    # Apply date filtering
                    if not self._is_date_in_range(last_date):
                        continue
                    
                    thread = iMessageThread(
                        chat_identifier=chat_identifier,
                        display_name=display_name or chat_identifier,
                        messages=messages[:self.config.max_messages_per_thread],
                        participant_count=len(set(msg['handle'] for msg in messages if msg['handle'])),
                        last_message_date=last_date,
                        total_messages=len(messages)
                    )
                    
                    threads.append(thread)
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading iMessage threads: {e}")
            raise
        
        return threads

    async def _load_messages_for_chat(self, cursor: sqlite3.Cursor, chat_identifier: str) -> List[Dict[str, Any]]:
        """Load messages for a specific chat."""
        message_query = """
        SELECT 
            m.text,
            m.date,
            m.is_from_me,
            h.id as handle,
            m.service as service_type,
            m.subject,
            m.balloon_bundle_id,
            m.associated_message_type
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        JOIN chat c ON cmj.chat_id = c.ROWID
        WHERE c.chat_identifier = ?
        AND m.text IS NOT NULL
        AND m.text != ''
        ORDER BY m.date ASC
        """
        
        cursor.execute(message_query, (chat_identifier,))
        message_results = cursor.fetchall()
        
        messages = []
        for msg_data in message_results:
            text, date, is_from_me, handle, service_type, subject, balloon_bundle_id, associated_message_type = msg_data
            
            # Skip special message types (reactions, edits, etc.)
            if associated_message_type and associated_message_type != 0:
                continue
                
            # Skip app-specific messages
            if balloon_bundle_id:
                continue
            
            message = {
                'text': text,
                'date': self._convert_core_data_timestamp(date),
                'is_from_me': bool(is_from_me),
                'handle': handle or 'Me',
                'service_type': service_type,
                'subject': subject
            }
            messages.append(message)
        
        return messages

    def _convert_core_data_timestamp(self, timestamp: int) -> datetime:
        """Convert Core Data timestamp to Python datetime."""
        # Core Data timestamps are seconds since 2001-01-01 00:00:00 UTC
        core_data_epoch = datetime(2001, 1, 1)
        return core_data_epoch + timedelta(seconds=timestamp)

    def _should_include_chat(self, chat_identifier: str, message_count: int) -> bool:
        """Determine if a chat should be included based on filtering criteria."""
        # Check message count
        if message_count < self.config.min_messages_per_thread:
            return False
        
        # Check chat type (group vs individual)
        is_group = '+' in chat_identifier or ';' in chat_identifier or 'chat' in chat_identifier.lower()
        
        if is_group and not self.config.include_group_chats:
            return False
        if not is_group and not self.config.include_individual_chats:
            return False
        
        # Check contact inclusion/exclusion
        if self.config.include_contacts:
            if not any(contact in chat_identifier for contact in self.config.include_contacts):
                return False
        
        if self.config.exclude_contacts:
            if any(contact in chat_identifier for contact in self.config.exclude_contacts):
                return False
        
        return True

    def _is_date_in_range(self, date: datetime) -> bool:
        """Check if date falls within the configured range."""
        if self.config.date_start:
            start_date = datetime.strptime(self.config.date_start, "%Y-%m-%d")
            if date < start_date:
                return False
        
        if self.config.date_end:
            end_date = datetime.strptime(self.config.date_end, "%Y-%m-%d")
            if date > end_date:
                return False
        
        return True

    def _organize_threads_by_bucket(self):
        """Organize threads by date buckets for efficient sampling."""
        for thread in self.threads:
            bucket = self._get_date_bucket(thread.last_message_date)
            if bucket not in self.threads_by_bucket:
                self.threads_by_bucket[bucket] = []
            self.threads_by_bucket[bucket].append(thread)
        
        # Log statistics
        for bucket in sorted(self.threads_by_bucket.keys(), reverse=True):
            count = len(self.threads_by_bucket[bucket])
            logger.info(f"Date bucket '{bucket}': {count} threads")

    def _get_date_bucket(self, date: datetime) -> str:
        """Get date bucket name for a given date."""
        bucket_start = date.replace(day=1)  # Start of month
        return bucket_start.strftime("%Y-%m")

    async def _sample_thread(self) -> iMessageThread:
        """Sample a thread with temporal bias."""
        if not self.threads_by_bucket:
            return random.choice(self.threads)
        
        # Sample bucket with temporal bias
        bucket = self._sample_date_bucket()
        
        # Sample thread from bucket
        return random.choice(self.threads_by_bucket[bucket])

    def _sample_date_bucket(self) -> str:
        """Sample a date bucket with exponential temporal decay."""
        buckets = list(self.threads_by_bucket.keys())
        
        if len(buckets) == 1:
            return buckets[0]
        
        # Calculate temporal weights
        weights = []
        current_time = datetime.now()
        
        for bucket in buckets:
            bucket_date = datetime.strptime(bucket, "%Y-%m")
            days_ago = (current_time - bucket_date).total_seconds() / (24 * 3600)
            
            # Calculate exponential decay weight
            if self.config.temporal_half_life_days is not None:
                decay_constant = math.log(2) / self.config.temporal_half_life_days
                weight = math.exp(-decay_constant * days_ago)
            else:
                weight = math.exp(-self.config.temporal_decay_rate * days_ago)
            
            # Weight by bucket size (dampened)
            bucket_size = len(self.threads_by_bucket[bucket])
            combined_weight = weight * (bucket_size ** 0.3)
            weights.append(combined_weight)
        
        return random.choices(buckets, weights=weights)[0]

    def _format_thread_content(self, thread: iMessageThread) -> str:
        """Format iMessage thread content for synthesis."""
        content_parts = [
            f"=== iMessage Conversation: {thread.display_name} ===",
            f"Participants: {thread.participant_count}",
            f"Total Messages: {thread.total_messages}",
            f"Date Range: {thread.messages[0]['date'].strftime('%Y-%m-%d')} to {thread.last_message_date.strftime('%Y-%m-%d')}",
            "",
            "Messages:"
        ]
        
        for i, msg in enumerate(thread.messages):
            sender = "Me" if msg['is_from_me'] else (msg['handle'] or "Unknown")
            timestamp = msg['date'].strftime('%Y-%m-%d %H:%M')
            content_parts.append(f"{timestamp} - {sender}: {msg['text']}")
        
        content_parts.append("=" * 50)
        
        return "\n".join(content_parts)


# Predefined prompts for iMessage synthesis
IMESSAGE_PROMPTS = [
    "You are analyzing an iMessage conversation. Please identify the main topic or purpose of this conversation and generate a question that tests understanding of what was discussed.",
    
    "You are analyzing an iMessage conversation. Please identify any important information, plans, or decisions made in this conversation and create a question about these key details.",
    
    "You are analyzing an iMessage conversation. Please identify the relationship dynamic between the participants and generate a question that tests understanding of their interaction style and context.",
    
    "You are analyzing an iMessage conversation. Please identify any problems, concerns, or issues discussed and create a question that tests understanding of these challenges.",
    
    "You are analyzing an iMessage conversation. Please generate a question that tests understanding of the overall context and outcome of this messaging exchange.",
    
    "You are analyzing an iMessage conversation. Please identify any emotional context, tone, or sentiment in the conversation and create a question about the interpersonal dynamics.",
    
    "You are analyzing an iMessage conversation. Please generate an instruction for an LLM to summarize the most important aspects of this conversation.",
    
    "You are analyzing an iMessage conversation. Please identify any future plans, commitments, or follow-up actions mentioned and create a question about these items.",
]