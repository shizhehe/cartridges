#!/usr/bin/env python3
"""
Interactive Tool Explorer for Code Memory Tools

This script allows you to explore and test tools configured in a synthesis config file.

Usage:
    python scratch/sabri/tool_explorer.py <config_file>
    
Example:
    python scratch/sabri/tool_explorer.py cartridges/configs/sabri/slack_synthesis.py
"""

import sys
import json
import asyncio
import importlib.util
import argparse
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel, ValidationError

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cartridges.data.tools import Tool, ToolSet, ToolInput, ToolOutput, instantiate_tools


class ToolExplorer:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.tools: Dict[str, Tool] = {}
        self.cleanup_functions: List[callable] = []
    
    async def load_tools_from_config(self) -> None:
        """Load tools from the provided config file."""
        print(f"🔍 Loading tools from config: {self.config_file}")
        
        try:
            # Load the config module
            spec = importlib.util.spec_from_file_location("config", self.config_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load config from {self.config_file}")
            
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Get the config object
            if not hasattr(config_module, 'config'):
                raise AttributeError("Config file must have a 'config' variable")
            
            synthesis_config = config_module.config
            
            # Extract tools from the synthesizer config
            if hasattr(synthesis_config, 'synthesizer') and hasattr(synthesis_config.synthesizer, 'tools'):
                tool_configs = synthesis_config.synthesizer.tools
                
                if tool_configs:
                    print(f"Found {len(tool_configs)} tool configs")
                    
                    # Instantiate the tools using the framework's function
                    tools, cleanup_funcs = await instantiate_tools(tool_configs)
                    
                    # Store tools by name
                    for tool in tools:
                        if isinstance(tool, Tool):
                            self.tools[tool.name] = tool
                            print(f"  ✅ Loaded tool: {tool.name}")
                        elif isinstance(tool, ToolSet):
                            # Handle ToolSet - extract individual tools
                            for individual_tool in tool.tools():
                                self.tools[individual_tool.name] = individual_tool
                                print(f"  ✅ Loaded tool from set: {individual_tool.name}")
                    
                    # Store cleanup functions
                    self.cleanup_functions.extend(cleanup_funcs)
                    
                else:
                    print("No tools found in config")
            else:
                print("Config does not have synthesizer.tools")
                
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            raise
    
    def list_tools(self) -> None:
        """List all available tools with their descriptions."""
        print("\n🔧 Available Tools:")
        print("=" * 50)
        
        if not self.tools:
            print("No tools found.")
            return
        
        for name, tool in self.tools.items():
            print(f"\n📋 {name}")
            print(f"   Description: {tool.description}")
            
            # Show input schema
            try:
                input_class = getattr(tool, "ToolInput", None)
                if input_class and hasattr(input_class, "model_json_schema"):
                    schema = input_class.model_json_schema()
                    properties = schema.get("properties", {})
                    if properties:
                        print("   Input fields:")
                        for field_name, field_info in properties.items():
                            field_type = field_info.get("type", "unknown")
                            field_desc = field_info.get("description", "")
                            default = field_info.get("default", "")
                            default_str = f" (default: {default})" if default else ""
                            print(f"     - {field_name}: {field_type}{default_str}")
                            if field_desc:
                                print(f"       {field_desc}")
            except Exception as e:
                print(f"   (Could not parse input schema: {e})")
    
    def get_tool_help(self, tool_name: str) -> None:
        """Show detailed help for a specific tool."""
        if tool_name not in self.tools:
            print(f"❌ Tool '{tool_name}' not found.")
            return
        
        tool = self.tools[tool_name]
        print(f"\n🔧 Tool: {tool_name}")
        print("=" * 50)
        print(f"Description: {tool.description}")
        
        # Show full tool definition
        try:
            definition = tool.definition
            print(f"\nTool Definition:")
            print(json.dumps(definition, indent=2))
        except Exception as e:
            print(f"Could not get tool definition: {e}")
        
        # Show example usage
        try:
            input_class = getattr(tool, "ToolInput", None)
            if input_class:
                schema = input_class.model_json_schema()
                properties = schema.get("properties", {})
                
                print(f"\nExample Usage:")
                example_input = {}
                for field_name, field_info in properties.items():
                    field_type = field_info.get("type", "string")
                    if field_type == "string":
                        example_input[field_name] = f"example_{field_name}"
                    elif field_type == "integer":
                        example_input[field_name] = 1
                    elif field_type == "boolean":
                        example_input[field_name] = True
                    elif field_type == "array":
                        example_input[field_name] = []
                
                print(f"Input: {json.dumps(example_input, indent=2)}")
        except Exception as e:
            print(f"Could not generate example: {e}")
    
    async def run_tool(self, tool_name: str, input_data: Dict[str, Any]) -> None:
        """Run a tool with the given input data."""
        if tool_name not in self.tools:
            print(f"❌ Tool '{tool_name}' not found.")
            return
        
        tool = self.tools[tool_name]
        
        try:
            # Create tool input
            input_class = getattr(tool, "ToolInput", ToolInput)
            tool_input = input_class(**input_data)
            
            print(f"🚀 Running tool '{tool_name}'...")
            print(f"Input: {json.dumps(input_data, indent=2)}")
            
            # Run the tool
            result = await tool.run_tool(tool_input)
            
            print(f"\n✅ Tool completed!")
            print(f"Success: {result.success}")
            if result.error:
                print(f"Error: {result.error}")
            if result.response:
                print(f"Response: {result.response}")
            
        except ValidationError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Error running tool: {e}")
    
    async def interactive_loop(self) -> None:
        """Run the interactive command loop."""
        print("\n🎯 Interactive Tool Explorer")
        print("=" * 50)
        print("Commands:")
        print("  list                 - List all available tools")
        print("  help <tool_name>     - Show detailed help for a tool")
        print("  run <tool_name>      - Run a tool (will prompt for input)")
        print("  exit                 - Exit the explorer")
        print()
        
        while True:
            try:
                command = input("🔧 tool-explorer> ").strip()
                
                if not command:
                    continue
                
                parts = command.split(None, 1)
                cmd = parts[0].lower()
                
                if cmd == "exit":
                    print("👋 Goodbye!")
                    break
                
                elif cmd == "list":
                    self.list_tools()
                
                elif cmd == "help":
                    if len(parts) < 2:
                        print("Usage: help <tool_name>")
                        continue
                    tool_name = parts[1]
                    self.get_tool_help(tool_name)
                
                elif cmd == "run":
                    if len(parts) < 2:
                        print("Usage: run <tool_name>")
                        continue
                    tool_name = parts[1]
                    
                    if tool_name not in self.tools:
                        print(f"❌ Tool '{tool_name}' not found. Use 'list' to see available tools.")
                        continue
                    
                    # Get input from user
                    print(f"\n🔧 Running tool: {tool_name}")
                    
                    try:
                        tool = self.tools[tool_name]
                        input_class = getattr(tool, "ToolInput", ToolInput)
                        schema = input_class.model_json_schema()
                        properties = schema.get("properties", {})
                        
                        if not properties:
                            # No input required
                            await self.run_tool(tool_name, {})
                        else:
                            print("Please provide input (JSON format):")
                            print("Example fields:")
                            for field_name, field_info in properties.items():
                                field_type = field_info.get("type", "unknown")
                                field_desc = field_info.get("description", "")
                                required = field_name in schema.get("required", [])
                                req_str = " (required)" if required else " (optional)"
                                print(f"  {field_name}: {field_type}{req_str}")
                                if field_desc:
                                    print(f"    {field_desc}")
                            
                            print("\nEnter JSON input (or 'cancel' to abort):")
                            input_str = input("JSON> ").strip()
                            
                            if input_str.lower() == "cancel":
                                continue
                            
                            try:
                                input_data = json.loads(input_str)
                                await self.run_tool(tool_name, input_data)
                            except json.JSONDecodeError as e:
                                print(f"❌ Invalid JSON: {e}")
                    
                    except Exception as e:
                        print(f"❌ Error setting up tool input: {e}")
                
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Available commands: list, help, run, exit")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all tools."""
        print("🧹 Cleaning up tools...")
        for cleanup_func in self.cleanup_functions:
            try:
                await cleanup_func()
            except Exception as e:
                print(f"Warning: cleanup failed: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive Tool Explorer")
    parser.add_argument("config_file", help="Path to the synthesis config file")
    
    args = parser.parse_args()
    config_file = Path(args.config_file)
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    explorer = ToolExplorer(config_file)
    
    try:
        # Load tools from config
        await explorer.load_tools_from_config()
        
        if not explorer.tools:
            print("❌ No tools loaded from config")
            sys.exit(1)
        
        # Start interactive loop
        await explorer.interactive_loop()
        
    finally:
        # Clean up
        await explorer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())