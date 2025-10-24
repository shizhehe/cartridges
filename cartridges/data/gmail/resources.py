
import os
import random
import asyncio
import math
from typing import List, Optional
from datetime import datetime, timedelta

from cartridges.data.gmail.download import GmailDownloader
from cartridges.data.resources import Resource
from cartridges.data.gmail.download import EmailThread
from .utils import get_service_pool
from cartridges.utils import get_logger

logger = get_logger(__name__)

class GmailResource(Resource):

    class Config(Resource.Config):

        # note: use the label as it appears in gmail search query (not the sidebar)
        labels: Optional[List[str]]=None

        # date format is YYYY/MM/DD
        date_start: Optional[str] = "2025/01/01"
        date_end: Optional[str] = None
        date_days_in_bucket: int = 30
        
        # Exponential decay parameters for temporal bias
        temporal_decay_rate: float = 0.1  # Higher = stronger recency bias
        temporal_half_life_days: Optional[int] = None  # Alternative to decay_rate
        
        # Additional Gmail search query to AND with label/date (e.g., 'from:"Allen Roberts"')
        search_query: Optional[str] = None

        cache_dir: str = os.path.expanduser("~/.cartridges/data/gmail")

    def __init__(self, config: Config):
        self.config = config

        self.downloader = GmailDownloader(
            cache_dir=self.config.cache_dir,
            pool_size=16,
            max_concurrent_downloads=3,
            batch_size=50,
            requests_per_minute=250,
            min_delay_between_requests=0.5,
        )

        # Get service pool for concurrent access
        self.service_pool = get_service_pool(pool_size=16)
        # Limit concurrent API calls to avoid rate limits
        self._api_semaphore = asyncio.Semaphore(16)  # Max 5 concurrent calls
    

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        """Sample a Gmail thread and return prompts for synthesis."""
        if not self.threads:
            raise ValueError("No threads loaded. Call setup() first.")
        
        # Sample a random thread
        sampled_thread = await self._sample_thread()
        
        # Get the thread content
        thread_content = await self._get_thread_content(sampled_thread)
        
        # Sample prompts from the predefined list
        prompts = random.choices(THREAD_PROMPTS, k=batch_size)
        
        return thread_content, prompts


    async def setup(self):
        """
        Fetch the metadata (ids) for all the threads in the user's gmail with 
        the labels specified in the config. Uses separate queries per label and date bucket.
        """
        self.threads = await self.downloader.get_threads(
            labels=self.config.labels,
            date_start=self.config.date_start,
            date_end=self.config.date_end,
            date_days_in_bucket=self.config.date_days_in_bucket,
            search_query=self.config.search_query,
        )

        # Initialize organization dictionaries
        self.threads_by_bucket = {}
        self.threads_by_label = {}
        
        # Organize threads by bucket for efficient sampling
        self._organize_threads_by_bucket()
        
        date_buckets = self._generate_date_buckets()
        logger.info(f"Loaded {len(self.threads)} Gmail threads across {len(date_buckets)} date buckets")
                
    
    def _generate_date_buckets(self) -> List[tuple]:
        """Generate date buckets based on config."""
        buckets = []
        
        if not self.config.date_start:
            # No date range specified, create a single "all time" bucket
            return [(None, None, "all_time")]
        
        start_date = datetime.strptime(self.config.date_start, "%Y/%m/%d")
        
        # If date_end is not provided, use today's date
        if self.config.date_end:
            end_date = datetime.strptime(self.config.date_end, "%Y/%m/%d")
        else:
            end_date = datetime.now().replace(hour=23, minute=59, second=59)  # End of today
        
        bucket_days = self.config.date_days_in_bucket
        
        current_date = start_date
        while current_date < end_date:
            bucket_end = min(current_date + timedelta(days=bucket_days), end_date)
            
            # Format dates for Gmail query (YYYY/MM/DD)
            bucket_start_str = current_date.strftime("%Y/%m/%d")
            bucket_end_str = bucket_end.strftime("%Y/%m/%d")
            
            # Create bucket name
            bucket_name = f"{current_date.strftime('%Y-%m-%d')}_to_{bucket_end.strftime('%Y-%m-%d')}"
            
            buckets.append((bucket_start_str, bucket_end_str, bucket_name))
            current_date = bucket_end
        
        logger.info(f"Generated {len(buckets)} date buckets of {bucket_days} days each")
        return buckets
    
  
    def _is_more_recent_bucket(self, bucket_a: str, bucket_b: str) -> bool:
        """Check if bucket_a is more recent than bucket_b."""
        if bucket_a == "all_time" or bucket_b == "all_time":
            return False  # Can't compare with all_time bucket
        
        try:
            # Extract start dates from bucket names (format: "2025-06-01_to_2025-06-30")
            date_a = bucket_a.split("_to_")[0]
            date_b = bucket_b.split("_to_")[0]
            return date_a > date_b  # String comparison works for YYYY-MM-DD format
        except:
            return False
    
    def _organize_threads_by_bucket(self):
        """Organize threads by label and date bucket for efficient sampling."""
        # Organize by both label and date bucket
        for thread in self.threads:
            label = thread.label
            bucket = thread.date_bucket
            
            # Add to bucket-based organization
            if bucket not in self.threads_by_bucket:
                self.threads_by_bucket[bucket] = []
            self.threads_by_bucket[bucket].append(thread)
            
            # Add to label-based organization
            if label not in self.threads_by_label:
                self.threads_by_label[label] = {}
            if bucket not in self.threads_by_label[label]:
                self.threads_by_label[label][bucket] = []
            self.threads_by_label[label][bucket].append(thread)
        
        # Log statistics
        logger.info("Thread organization:")
        for label in sorted(self.threads_by_label.keys()):
            total_in_label = sum(len(buckets) for buckets in self.threads_by_label[label].values())
            logger.info(f"Label '{label}': {total_in_label} threads")
            for bucket in sorted(self.threads_by_label[label].keys(), reverse=True):
                count = len(self.threads_by_label[label][bucket])
                logger.info(f"  {bucket}: {count} threads")
        
        logger.info(f"Organized {len(self.threads)} threads into {len(self.threads_by_label)} labels × {len(self.threads_by_bucket)} date buckets")

    
    async def _sample_thread(self) -> EmailThread:
        """Sample a random thread with label and recency bias applied at sampling time."""
        if not self.threads_by_label:
            raise ValueError("No threads loaded. Call setup() first.")
        
        # (1) Sample a label with weight bias
        label = self._sample_label()
        
        # (2) Sample a date bucket with recency bias for that label
        bucket = self._sample_date_bucket_for_label(label)
        
        # (3) Sample a thread from that label×bucket combination
        bucket_threads = self.threads_by_label[label][bucket]
        return random.choice(bucket_threads)
    
    def _sample_label(self) -> str:
        """Sample a label based on configured weights."""
        if not self.config.labels:
            # No label configuration, return first available label
            return list(self.threads_by_label.keys())[0]
        
        # Create weighted list of labels
        labels = []
        weights = []
        
        for label in self.config.labels:
            if label in self.threads_by_label:
                labels.append(label)
                weights.append(1.0)  # Equal weight since config.labels is now just strings
        
        if not labels:
            # Fallback to any available label
            return list(self.threads_by_label.keys())[0]
        
        # Use weighted random selection
        return random.choices(labels, weights=weights)[0]

    def _sample_date_bucket_for_label(self, label: str) -> str:
        """Sample a date bucket with exponential temporal decay for a specific label."""
        if label not in self.threads_by_label:
            raise ValueError(f"Label '{label}' not found in threads")
        
        buckets = list(self.threads_by_label[label].keys())
        
        if len(buckets) == 1:
            return buckets[0]
        
        # Calculate temporal weights using exponential decay
        weights = []
        current_time = datetime.now()
        
        for bucket in buckets:
            bucket_size = len(self.threads_by_label[label][bucket])
            
            # Calculate days since bucket start date
            days_ago = self._get_days_since_bucket(bucket, current_time)
            
            # Calculate exponential decay weight
            temporal_weight = self._calculate_temporal_weight(days_ago)
            
            # Combine temporal weight with bucket size (dampened)
            combined_weight = temporal_weight * (bucket_size ** 0.3)  # Light bucket size influence
            weights.append(combined_weight)
        
        # Use weighted random selection
        return random.choices(buckets, weights=weights)[0]
    
    def _get_days_since_bucket(self, bucket_name: str, current_time: datetime) -> float:
        """Calculate days since the start of a date bucket."""
        if bucket_name == "all_time":
            return 365.0  # Treat as very old
        
        try:
            # Extract start date from bucket name (format: "2025-06-01_to_2025-06-30")
            bucket_start_str = bucket_name.split("_to_")[0]
            bucket_start = datetime.strptime(bucket_start_str, "%Y-%m-%d")
            days_ago = (current_time - bucket_start).total_seconds() / (24 * 3600)
            return max(0, days_ago)  # Ensure non-negative
        except (ValueError, IndexError):
            return 365.0  # Default to old if parsing fails
    
    def _calculate_temporal_weight(self, days_ago: float) -> float:
        """Calculate exponential decay weight based on days ago."""
        if self.config.temporal_half_life_days is not None:
            # Use half-life approach: weight = exp(-ln(2) * days_ago / half_life)
            decay_constant = math.log(2) / self.config.temporal_half_life_days
            return math.exp(-decay_constant * days_ago)
        else:
            # Use decay rate approach: weight = exp(-decay_rate * days_ago)
            return math.exp(-self.config.temporal_decay_rate * days_ago)

    async def _get_thread_content(self, thread: EmailThread) -> str:
        """Get the full content of a Gmail thread using minimal fields."""       
        thread_content = []
        for message in thread.messages:
            thread_content.append(f"From: {message.from_addr}\nTo: {message.to_addr}\nSubject: {message.subject}\n\n{message.body}\n" + "="*50)
        
        return "\n\n".join(thread_content)
            
    


THREAD_PROMPTS = [
    (
        "You are analyzing an email thread. Please identify a key insight, decision, or important information shared in the conversation and generate a question that can be used to test understanding of this insight."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify something the user learned or discovered from the conversation and generate a question that can be used to test understanding of this lesson."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify a key challenge, problem, or difficulty discussed in the conversation and generate a question that tests understanding of this issue."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify the main topic or subject of the conversation and generate a comprehensive question that tests understanding of the discussion."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please generate a question that tests understanding of the overall context and outcome of the email conversation."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),
    (
        "You are analyzing an email thread. Please generate a single instruction for an LLM to summarize the most important part of this email conversation."
    )
]


