
import asyncio
import os
import base64
import hashlib
import time
import random
from typing import Optional, List, Dict, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from pydrantic import BaseConfig
from cartridges.data.gmail.utils import get_service_pool
from cartridges.utils import get_logger

logger = get_logger(__name__)

# TODOS: 
# [] Get rid of the labelconfig 
@dataclass
class EmailMessage:
    id: str
    from_addr: str
    to_addr: str
    date: str
    subject: str
    body: str
    snippet: str


@dataclass
class EmailThread:
    """Structure for a downloaded Gmail thread."""
    id: str
    label: str
    date_bucket: str
    subject: str
    participants: List[str]
    messages: List[EmailMessage]
    content_hash: str
    last_modified: str
    thread_size: int  # Number of messages in thread
    history_id: str = ""  # Gmail history ID for change detection


@dataclass
class ThreadMetadata:
    id: str
    label: str
    date_bucket: str  # Format: "2025-06-01_to_2025-06-30"


class DownloadGmailConfig(BaseConfig):
    # note: use the label as it appears in gmail search query (not the sidebar)
    labels: Optional[List[str]] = None

    # date format is YYYY/MM/DD
    date_start: Optional[str] = "2025/09/01"
    date_end: Optional[str] = "2025/09/14"
    date_days_in_bucket: int = 30

    cache_dir: str = os.path.expanduser("~/.cartridges/data/gmail")
    
    # Performance settings
    max_concurrent_downloads: int = 3  # Reduced for rate limiting
    batch_size: int = 50  # Reduced batch size
    requests_per_minute: int = 250  # Gmail API limit
    min_delay_between_requests: float = 0.5  # Minimum delay in seconds

    # Extra Gmail search query ANDed with label/date (e.g., 'from:"Alun Roberts"')
    search_query: Optional[str] = None

    def run(self):
        return asyncio.run(main(self))


async def main(config: DownloadGmailConfig):
    """Download Gmail threads and store them in parquet format."""
    logger.info("Starting Gmail download process")
    
    # Ensure destination directory exists
    os.makedirs(config.cache_dir, exist_ok=True)
    
    downloader = GmailDownloader(
        cache_dir=config.cache_dir,
        pool_size=16,
        max_concurrent_downloads=3,
        batch_size=50,
        requests_per_minute=250,
        min_delay_between_requests=0.5,
    )
    await downloader.get_threads(
        labels=config.labels,
        date_start=config.date_start,
        date_end=config.date_end,
        date_days_in_bucket=config.date_days_in_bucket,
        search_query=config.search_query,
    )
    
    logger.info("Gmail download completed successfully")


class GmailDownloader:
    """Handles downloading Gmail threads with deduplication and incremental updates."""
    
    def __init__(
        self, 
        cache_dir: str,

        pool_size: int = 16,
        max_concurrent_downloads: int = 3,  # Reduced for rate limiting
        batch_size: int = 50,  # Reduced batch size
        requests_per_minute: int = 250,  # Gmail API limit
        min_delay_between_requests: float = 0.5  # Minimum delay in seconds
    ):
        self.service_pool = get_service_pool(pool_size=pool_size)
        self._api_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.parquet_file = Path(cache_dir) / "gmail_threads.parquet"
        self.existing_threads: Dict[str, EmailThread] = {}

        # Store configuration parameters
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.min_delay_between_requests = min_delay_between_requests
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()
    
    async def get_threads(
        self,
        labels: Optional[List[str]] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        date_days_in_bucket: int = 30,
        search_query: Optional[str] = None,
    ) -> List[EmailThread]:
        """Main download orchestration method."""
        # Store parameters as instance variables for use in other methods
        self.labels = labels
        self.date_start = date_start
        self.date_end = date_end
        self.date_days_in_bucket = date_days_in_bucket
        self.search_query = search_query
        
        logger.info("Loading existing downloaded threads")
        await self._load_existing_threads()
        
        logger.info("Fetching thread metadata from Gmail")
        thread_metadata = await self._fetch_all_thread_metadata()
        
        logger.info("Identifying threads to download/update")
        threads_to_process = await self._identify_threads_to_process(thread_metadata)
        
        if not threads_to_process:
            logger.info("All threads are up to date")
            return list(self.existing_threads.values())
        
        logger.info(f"Downloading {len(threads_to_process)} threads")
        downloaded_threads = await self._download_thread_contents(threads_to_process)
        
        logger.info("Saving threads to parquet")
        # Combine existing and new threads
        self.existing_threads.update({thread.id: thread for thread in downloaded_threads})
        final_threads = list(self.existing_threads.values())
        
        # Convert to records for parquet using asdict
        records = [asdict(thread) for thread in final_threads]
        
        # Write to parquet
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.parquet_file, compression="snappy")
        
        logger.info(f"Saved {len(final_threads)} threads to {self.parquet_file}")
        return final_threads
    
    async def _wait_for_rate_limit(self):
        """Implement rate limiting to respect Gmail API quotas."""
        current_time = time.time()
        
        # Reset request count if we're in a new minute window
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time
        
        # If we've hit the rate limit, wait until next minute
        if self._request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self._request_window_start)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._request_window_start = time.time()
        
        # Enforce minimum delay between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.min_delay_between_requests:
            delay = self.min_delay_between_requests - time_since_last
            await asyncio.sleep(delay)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    async def _make_api_request_with_retry(self, api_call, max_retries: int = 5):
        """Make an API request with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                
                async with self._api_semaphore:
                    service = await self.service_pool.get_service_async()
                    try:
                        result = await asyncio.to_thread(api_call, service)
                        return result
                    finally:
                        await self.service_pool.return_service_async(service)
                        
            except Exception as e:
                if "rateLimitExceeded" in str(e) or "Quota exceeded" in str(e):
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error, re-raise
                    raise e
        
        raise Exception(f"Failed after {max_retries} attempts due to rate limiting")
        
    async def _load_existing_threads(self):
        """Load existing downloaded threads from parquet file."""
        if not self.parquet_file.exists():
            logger.info("No existing parquet file found")
            return
        
        try:
            df = pd.read_parquet(self.parquet_file)
            records = df.to_dict('records')
            
            self.existing_threads = {}
            for row in records:
                # Convert message dictionaries back to EmailMessage objects
                messages = [
                    EmailMessage(**msg) if isinstance(msg, dict) else msg
                    for msg in row['messages']
                ]
                row['messages'] = messages
                self.existing_threads[row['id']] = EmailThread(**row)
                
            logger.info(f"Loaded {len(self.existing_threads)} existing threads")
        except Exception as e:
            logger.warning(f"Failed to load existing threads: {e}")
            self.existing_threads = {}
    
    async def _fetch_all_thread_metadata(self):
        """Fetch metadata for all threads matching the criteria."""
        date_buckets = self._generate_date_buckets()
        fetch_tasks = []
        
        if not self.labels:
            # No labels specified, fetch all threads for each date bucket
            for bucket_start, bucket_end, bucket_name in date_buckets:
                task = self._fetch_thread_metadata_for_label_and_date(
                    None, bucket_start, bucket_end, bucket_name
                )
                fetch_tasks.append(task)
        else:
            # Fetch threads for each label and date bucket combination
            for label in self.labels:
                for bucket_start, bucket_end, bucket_name in date_buckets:
                    task = self._fetch_thread_metadata_for_label_and_date(
                        label, bucket_start, bucket_end, bucket_name
                    )
                    fetch_tasks.append(task)
        
        logger.info(f"Starting {len(fetch_tasks)} metadata fetch operations")
        task_results = await asyncio.gather(*fetch_tasks)
        
        # Flatten results
        all_metadata = []
        for thread_list in task_results:
            if thread_list:
                all_metadata.extend(thread_list)
        
        logger.info(f"Found {len(all_metadata)} threads total")
        return all_metadata
    
    def _generate_date_buckets(self) -> List[tuple]:
        """Generate date buckets based on config."""
        buckets = []
        
        if not self.date_start:
            return [(None, None, "all_time")]
        
        start_date = datetime.strptime(self.date_start, "%Y/%m/%d")
        
        if self.date_end:
            end_date = datetime.strptime(self.date_end, "%Y/%m/%d")
        else:
            end_date = datetime.now().replace(hour=23, minute=59, second=59)
        
        bucket_days = self.date_days_in_bucket
        current_date = start_date
        
        while current_date < end_date:
            bucket_end = min(current_date + timedelta(days=bucket_days), end_date)
            
            bucket_start_str = current_date.strftime("%Y/%m/%d")
            bucket_end_str = bucket_end.strftime("%Y/%m/%d")
            bucket_name = f"{current_date.strftime('%Y-%m-%d')}_to_{bucket_end.strftime('%Y-%m-%d')}"
            
            buckets.append((bucket_start_str, bucket_end_str, bucket_name))
            current_date = bucket_end
        
        return buckets
    
    async def _fetch_thread_metadata_for_label_and_date(
        self, label_name: Optional[str], date_start: Optional[str], 
        date_end: Optional[str], bucket_name: str
    ) -> List[Dict]:
        """Fetch thread metadata for a specific label and date range."""
        query_parts = []
        
        if label_name:
            query_parts.append(f"label:{label_name}")
        if date_start:
            query_parts.append(f"after:{date_start}")
        if date_end:
            query_parts.append(f"before:{date_end}")
        if self.search_query:
            query_parts.append(self.search_query)
        
        query = " ".join(query_parts) if query_parts else ""
        label_display = label_name if label_name else "ALL"
        
        logger.info(f"Fetching {label_display} threads in {bucket_name}")
        
        page_token = None
        collected_metadata = []
        
        while True:
            request_params = {
                'q': query,
                'fields': 'threads(id,historyId,snippet),nextPageToken',
                'maxResults': 1000
            }
            
            if page_token:
                request_params['pageToken'] = page_token
            
            result = await self._make_api_request_with_retry(
                lambda service: service.users().threads().list(
                    userId='me', **request_params
                ).execute()
            )
            
            threads = result.get('threads', [])
            
            for thread in threads:
                collected_metadata.append({
                    'id': thread['id'],
                    'label': label_name or "ALL",
                    'date_bucket': bucket_name,
                    'history_id': thread.get('historyId', '')
                })
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        logger.info(f"Found {len(collected_metadata)} threads")
        return collected_metadata
    
    async def _identify_threads_to_process(self, thread_metadata: List[Dict]) -> List[Dict]:
        """Identify which threads need to be downloaded or updated."""
        threads_to_process = []
        
        for metadata in thread_metadata:
            thread_id = metadata['id']
            history_id = metadata.get('history_id', '')
            
            if thread_id not in self.existing_threads:
                # New thread, needs downloading
                threads_to_process.append(metadata)
            else:
                existing_thread = self.existing_threads[thread_id]
                # Check if thread was modified using history_id
                if existing_thread.history_id != history_id:
                    logger.info(f"Thread {thread_id} was modified, will re-download")
                    threads_to_process.append(metadata)
        
        logger.info(f"Need to process {len(threads_to_process)} threads")
        return threads_to_process
    
    async def _download_thread_contents(self, thread_metadata: List[Dict]) -> List[EmailThread]:
        """Download full content for the specified threads."""
        downloaded_threads = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(thread_metadata), self.batch_size):
            batch = thread_metadata[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch)} threads)")
            
            # Download batch concurrently
            download_tasks = [
                self._download_single_thread(metadata) 
                for metadata in batch
            ]
            
            batch_results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Collect successful downloads
            for result in batch_results:
                if isinstance(result, EmailThread):
                    downloaded_threads.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error downloading thread: {result}")
            
            # Add delay between batches to be extra conservative
            if i + self.batch_size < len(thread_metadata):
                logger.info("Waiting 2s before next batch")
                await asyncio.sleep(2)
        
        return downloaded_threads
    
    async def _download_single_thread(self, metadata: Dict) -> EmailThread:
        """Download a single thread's content."""
        thread_id = metadata['id']
        
        # Get full thread content with retry logic
        thread = await self._make_api_request_with_retry(
            lambda service: service.users().threads().get(
                userId='me',
                id=thread_id,
                fields='messages(id,payload,snippet,internalDate,historyId)'
            ).execute()
        )
        
        # Extract thread information
        messages = thread.get('messages', [])
        
        if not messages:
            raise ValueError(f"Thread {thread_id} has no messages")
        
        # Extract subject and participants
        first_message = messages[0]
        headers = first_message.get('payload', {}).get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        
        # Collect all unique participants and find the latest message time
        participants = set()
        processed_messages = []
        latest_internal_date = 0
        
        for message in messages:
            payload = message.get('payload', {})
            msg_headers = payload.get('headers', [])
            
            # Extract sender and recipient info
            from_addr = next((h['value'] for h in msg_headers if h['name'] == 'From'), '')
            to_addr = next((h['value'] for h in msg_headers if h['name'] == 'To'), '')
            
            if from_addr:
                participants.add(from_addr)
            if to_addr:
                participants.update(addr.strip() for addr in to_addr.split(','))
            
            # Extract message content
            body = self._extract_message_body(payload)
            
            # Track the latest message timestamp
            internal_date = int(message.get('internalDate', '0'))
            if internal_date > latest_internal_date:
                latest_internal_date = internal_date
            
            processed_messages.append(EmailMessage(
                id=message['id'],
                from_addr=from_addr,
                to_addr=to_addr,
                date=message.get('internalDate', ''),
                subject=next((h['value'] for h in msg_headers if h['name'] == 'Subject'), ''),
                body=body,
                snippet=message.get('snippet', '')
            ))
        
        # Convert latest timestamp to ISO format
        if latest_internal_date > 0:
            # Gmail internalDate is in milliseconds since epoch
            last_modified = datetime.fromtimestamp(latest_internal_date / 1000).isoformat()
        else:
            # Fallback to current time if no valid timestamp found
            last_modified = datetime.now().isoformat()
        
        # Create content hash for deduplication
        content_for_hash = f"{subject}|{sorted(participants)}|{len(processed_messages)}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
        
        return EmailThread(
            id=thread_id,
            label=metadata['label'],
            date_bucket=metadata['date_bucket'],
            subject=subject,
            participants=list(participants),
            messages=processed_messages,
            content_hash=content_hash,
            last_modified=last_modified,
            thread_size=len(processed_messages),
            history_id=metadata.get('history_id', '')
        )
    
    def _extract_message_body(self, payload: dict) -> str:
        """Extract text from Gmail message payload (supports nested MIME and HTML)."""
        import base64
        import re

        def html_to_text(html: str) -> str:
            text = re.sub(r"<\s*br\s*/?>", "\n", html, flags=re.IGNORECASE)
            text = re.sub(r"<\s*/p\s*>", "\n\n", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def decode_part_data(data: str) -> str:
            if not data:
                return ""
            try:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
            except Exception:
                return ""

        def walk(node: dict) -> tuple[list[str], list[str]]:
            plain_texts: list[str] = []
            html_texts: list[str] = []
            if not isinstance(node, dict):
                return plain_texts, html_texts

            mime = node.get("mimeType", "") or ""

            if "parts" in node and node.get("parts"):
                for child in node["parts"]:
                    p, h = walk(child)
                    plain_texts.extend(p)
                    html_texts.extend(h)
            else:
                body_data = node.get("body", {}).get("data", "")
                if not body_data:
                    return plain_texts, html_texts
                decoded = decode_part_data(body_data)
                if not decoded:
                    return plain_texts, html_texts

                if mime.startswith("text/plain"):
                    plain_texts.append(decoded.strip())
                elif mime.startswith("text/html"):
                    html_texts.append(html_to_text(decoded))

            return plain_texts, html_texts

        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        plains, htmls = walk(payload)

        # Prefer the longest plain text body if present; otherwise use the longest HTML-to-text body
        candidate = None
        if plains:
            # Deduplicate by normalized content and choose the longest
            unique_plains: dict[str, str] = {}
            for t in plains:
                unique_plains[normalize(t)] = t.strip()
            candidate = max(unique_plains.values(), key=len)
        elif htmls:
            unique_htmls: dict[str, str] = {}
            for t in htmls:
                unique_htmls[normalize(t)] = t.strip()
            candidate = max(unique_htmls.values(), key=len)

        return candidate.strip() if candidate else "No readable content"
    



if __name__ == "__main__":
    DownloadGmailConfig(
        labels=[
            # "categories--stanford--primary--stanford-"
            # LabelConfig(name="categories--stanford--updates--stanford-", weight=1.0),
        ],
        date_start="2025/09/01",
        date_end="2025/09/14",
        date_days_in_bucket=30,
    ).run()