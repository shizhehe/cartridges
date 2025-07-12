#!/usr/bin/env python3
"""
Test script for Gmail Resource

This script tests the Gmail resource implementation with breakpoints for debugging.

Usage:
    python scratch/sabri/test_gmail_resource.py
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cartridges.data.gmail.resources import GmailResource, LabelConfig


async def test_gmail_resource():
    """Test the Gmail resource setup and sampling."""
    
    print("🧪 Testing Gmail Resource")
    print("=" * 50)
    
    # Check environment setup
    print("🔍 Checking environment setup...")
    if "CARTRIDGES_DIR" not in os.environ:
        print("❌ CARTRIDGES_DIR environment variable not set!")
        print("💡 Setting it automatically for this test...")
        os.environ["CARTRIDGES_DIR"] = str(project_root)
        print(f"   CARTRIDGES_DIR={project_root}")
    
    cartridges_dir = os.environ["CARTRIDGES_DIR"]
    secrets_dir = os.path.join(cartridges_dir, "secrets")
    
    print(f"📁 CARTRIDGES_DIR: {cartridges_dir}")
    print(f"📁 Secrets directory: {secrets_dir}")
    
    if not os.path.exists(secrets_dir):
        print("❌ Secrets directory not found!")
        print("💡 Please create it and add your Gmail credentials:")
        print(f"   mkdir -p {secrets_dir}")
        print(f"   # Add credentials.json to {secrets_dir}/")
        return
    
    credentials_path = os.path.join(secrets_dir, "credentials.json")
    if not os.path.exists(credentials_path):
        print("❌ Gmail credentials.json not found!")
        print("💡 Please follow the setup instructions:")
        print("   1. Go to https://console.cloud.google.com/")
        print("   2. Create a project and enable Gmail API")
        print("   3. Create OAuth 2.0 credentials (Desktop app)")
        print(f"   4. Download and save as {credentials_path}")
        return
    
    print("✅ Environment setup looks good")
    
    # Configure Gmail resource with some common labels
    config = GmailResource.Config(
        labels=[
            # LabelConfig(name="Categories (stanford)/Primary (stanford)", weight=1.0),
            # LabelConfig(name="Categories (stanford)/Updates (stanford)", weight=2.0),  # Higher weight for important emails
            LabelConfig(name="categories--stanford--primary--stanford-", weight=1.0),
            LabelConfig(name="categories--stanford--updates--stanford-", weight=1.0),
            # You can add more labels here like:
            # LabelConfig(name="WORK", weight=1.5),
            # LabelConfig(name="PERSONAL", weight=1.0),
        ]
    )
    
    # Create resource instance
    gmail_resource = GmailResource(config)
    
    # Breakpoint 1: After creating resource
    breakpoint()  # Check gmail_resource configuration
    
    try:
        print("\n📧 Setting up Gmail resource...")
        await gmail_resource.setup()
        
        # Breakpoint 2: After setup completes
        breakpoint()  # Check loaded threads: len(gmail_resource.threads), gmail_resource.threads[:5]
        
        print(f"\n✅ Setup complete! Loaded {len(gmail_resource.threads)} threads")
        
        # Test sampling
        print("\n🎲 Testing sampling...")
        batch_size = 3
        
        thread_content, prompts = await gmail_resource.sample_prompt(batch_size)
        
        # Breakpoint 3: After sampling
        breakpoint()  # Examine thread_content and prompts
        
        print(f"\n📋 Sampled content length: {len(thread_content)} characters")
        print(f"📝 Generated {len(prompts)} prompts")
        
        # Display first 500 characters of content
        print(f"\n📄 Content preview:")
        print("-" * 50)
        print(thread_content[:500] + "..." if len(thread_content) > 500 else thread_content)
        
        print(f"\n💭 Sample prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt[:100]}...")
        
        # Test multiple samples
        print("\n🔄 Testing multiple samples...")
        for i in range(3):
            sampled_thread = await gmail_resource._sample_thread()
            print(f"Sample {i+1}: Thread ID {sampled_thread.id}, Label: {sampled_thread.label}, Bucket: {sampled_thread.date_bucket}")
            
            # Breakpoint 4: Check individual thread samples
            if i == 0:  # Only break on first iteration
                breakpoint()  # Examine sampled_thread details
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        breakpoint()  # Debug the error
        raise


def main():
    """Main entry point."""
    asyncio.run(test_gmail_resource())


if __name__ == "__main__":
    main()