from typing import Dict, List, Optional, Tuple, Any
import json
import email
from pathlib import Path
from pydantic import BaseModel
from collections import defaultdict
import re
from datetime import datetime
import pandas as pd


class EnronEmail(BaseModel):
    email_id: str
    from_addr: str
    to_addr: str
    subject: str
    date: str
    folder: str
    body: str
    message_id: Optional[str] = None
    cc: Optional[str] = None
    user_id: str  # Which user this email belongs to
    file_path: str  # Path to original email file
    parsed_date: Optional[datetime] = None  # Parsed datetime object


class EnronUser(BaseModel):
    user_id: str
    name: str
    emails: Dict[str, EnronEmail]
    folders: List[str]
    total_emails: int = 0


def extract_emails_from_string(email_string: str) -> List[str]:
    """Extract email addresses from a string."""
    if not email_string:
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, email_string)
    return emails


def parse_enron_email_file(email_path: Path, user_id: str) -> Optional[EnronEmail]:
    """Parse a single Enron email file."""
    try:
        with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse email using email library
        msg = email.message_from_string(content)
        
        # Extract body
        body = ""
        if msg.is_multipart():
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(payload.decode('utf-8', errors='ignore'))
                        else:
                            body_parts.append(part.get_payload())
                    except:
                        continue
            body = '\n'.join(body_parts)
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
                else:
                    body = msg.get_payload()
            except:
                body = str(msg.get_payload())
        
        # Parse date
        parsed_date = None
        date_str = msg.get('Date', '')
        if date_str:
            try:
                parsed_date = email.utils.parsedate_to_datetime(date_str)
            except:
                pass
        
        # Create email object
        enron_email = EnronEmail(
            email_id=email_path.stem,
            from_addr=msg.get('From', ''),
            to_addr=msg.get('To', ''),
            subject=msg.get('Subject', ''),
            date=date_str,
            folder=email_path.parent.name,
            body=body.strip(),
            message_id=msg.get('Message-ID', ''),
            cc=msg.get('Cc', ''),
            user_id=user_id,
            file_path=str(email_path),
            parsed_date=parsed_date
        )
        
        return enron_email
        
    except Exception as e:
        print(f"Error parsing {email_path}: {e}")
        return None


def load_enron_user_data(
    user_ids: Optional[List[str]] = None,
    dataset_path: str = "datasets/enron",
    folders_to_include: Optional[List[str]] = None,
    max_emails_per_user: Optional[int] = None
) -> List[EnronUser]:
    """Load Enron email data for specified users."""
    
    base_path = Path(dataset_path) / "raw" / "maildir"
    
    if not base_path.exists():
        raise ValueError(f"Enron dataset not found at {base_path}")
    
    # Get available users
    available_users = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    # Filter users if specified
    if user_ids:
        users_to_process = [u for u in user_ids if u in available_users]
        if not users_to_process:
            raise ValueError(f"None of the specified users found in dataset: {user_ids}")
    else:
        users_to_process = available_users[:10]  # Default to first 10 users
    
    print(f"Loading data for {len(users_to_process)} users...")
    
    users = []
    
    for user_id in users_to_process:
        user_path = base_path / user_id
        
        # Get folders for this user
        user_folders = [d.name for d in user_path.iterdir() if d.is_dir()]
        
        # Filter folders if specified
        if folders_to_include:
            folders_to_scan = [f for f in user_folders if f in folders_to_include]
        else:
            folders_to_scan = user_folders
        
        # Load emails from specified folders
        user_emails = {}
        email_count = 0
        
        for folder in folders_to_scan:
            folder_path = user_path / folder
            if not folder_path.exists():
                continue
                
            # Get all email files in this folder
            email_files = [f for f in folder_path.iterdir() if f.is_file()]
            
            for email_file in email_files:
                if max_emails_per_user and email_count >= max_emails_per_user:
                    break
                    
                enron_email = parse_enron_email_file(email_file, user_id)
                if enron_email:
                    # Use unique key: folder_filename
                    email_key = f"{folder}_{email_file.name}"
                    user_emails[email_key] = enron_email
                    email_count += 1
            
            if max_emails_per_user and email_count >= max_emails_per_user:
                break
        
        # Create user object
        enron_user = EnronUser(
            user_id=user_id,
            name=user_id.replace('-', ' ').title(),  # Simple name conversion
            emails=user_emails,
            folders=folders_to_scan,
            total_emails=len(user_emails)
        )
        
        users.append(enron_user)
        print(f"Loaded {len(user_emails)} emails for user {user_id}")
    
    print(f"Successfully loaded data for {len(users)} users")
    return users


def create_temporal_batches(
    users: List[EnronUser], 
    max_emails_per_batch: int = 1000,
    num_batches: Optional[int] = None
) -> Tuple[List[List[Tuple[str, EnronEmail]]], List[Dict[str, Any]]]:
    """Create temporal batches from user email data.
    
    Returns:
        batches: List of batches, each containing (file_path, email) tuples
        metadata: List of metadata dicts for each batch
    """
    
    # Collect all emails with valid dates
    all_emails_with_dates = []
    
    for user in users:
        for email_key, email in user.emails.items():
            if email.parsed_date:
                all_emails_with_dates.append((email.file_path, email))
    
    print(f"Found {len(all_emails_with_dates)} emails with valid dates")
    
    if not all_emails_with_dates:
        return [], []
    
    # Sort by date
    all_emails_with_dates.sort(key=lambda x: x[1].parsed_date)
    
    # If num_batches specified, limit the data to fit exactly that many batches
    if num_batches:
        total_emails = len(all_emails_with_dates)
        max_emails_to_use = num_batches * max_emails_per_batch
        
        if max_emails_to_use < total_emails:
            print(f"Using first {max_emails_to_use} emails out of {total_emails} to create exactly {num_batches} batches of size {max_emails_per_batch}")
            all_emails_with_dates = all_emails_with_dates[:max_emails_to_use]
        else:
            print(f"Using all {total_emails} emails to create up to {num_batches} batches of size {max_emails_per_batch}")
    
    # Create batches
    batches = []
    batch_metadata = []
    
    current_batch = []
    batch_start_date = None
    batch_end_date = None
    
    for file_path, email in all_emails_with_dates:
        if not current_batch:
            batch_start_date = email.parsed_date
        
        current_batch.append((file_path, email))
        batch_end_date = email.parsed_date
        
        # Check if batch is full or if we've reached the target number of batches
        batch_is_full = len(current_batch) >= max_emails_per_batch
        reached_target_batches = num_batches and len(batches) >= num_batches
        
        if batch_is_full and not reached_target_batches:
            # Finalize current batch
            batches.append(current_batch.copy())
            
            metadata = {
                'batch_id': len(batches) - 1,
                'num_emails': len(current_batch),
                'start_date': batch_start_date,
                'end_date': batch_end_date,
                'email_paths': [fp for fp, _ in current_batch],
                'users': list(set(email.user_id for _, email in current_batch))
            }
            batch_metadata.append(metadata)
            
            # Reset for next batch
            current_batch = []
            batch_start_date = None
            batch_end_date = None
    
    # Handle remaining emails in final batch
    if current_batch:
        batches.append(current_batch)
        
        metadata = {
            'batch_id': len(batches) - 1,
            'num_emails': len(current_batch),
            'start_date': batch_start_date,
            'end_date': batch_end_date,
            'email_paths': [fp for fp, _ in current_batch],
            'users': list(set(email.user_id for _, email in current_batch))
        }
        batch_metadata.append(metadata)
    
    print(f"Created {len(batches)} temporal batches")
    
    return batches, batch_metadata


def load_enron_selected_users(
    selected_users_path: str = "datasets/enron/processed/selected_users.json",
    dataset_path: str = "datasets/enron",
    max_emails_per_batch: Optional[int] = None,
    num_batches: Optional[int] = None
) -> List[EnronUser]:
    """Load data for pre-selected Enron users with their specified folders."""
    
    try:
        with open(selected_users_path, 'r') as f:
            users_config = json.load(f)
    except FileNotFoundError:
        print(f"Selected users file not found: {selected_users_path}")
        return []
    
    users = []
    
    # Calculate max emails per user if batch parameters are provided
    max_emails_per_user = None
    if max_emails_per_batch and num_batches:
        max_emails_per_user = max_emails_per_batch * num_batches
    
    for user_id, folder_config in users_config.items():
        try:
            # folder_config is a dict like {"inbox": "Inbox", "sent": "Sent", ...}
            folders_to_include = list(folder_config.keys())
            
            user_data = load_enron_user_data(
                user_ids=[user_id],
                dataset_path=dataset_path,
                folders_to_include=folders_to_include,
                max_emails_per_user=max_emails_per_user
            )
            
            if user_data:
                users.extend(user_data)
                
        except Exception as e:
            print(f"Error loading user {user_id}: {e}")
            continue
    
    return users