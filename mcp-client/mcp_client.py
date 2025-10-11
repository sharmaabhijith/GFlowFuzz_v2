#!/usr/bin/env python3

import asyncio
import json
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@dataclass
class ToolResult:
    tool_name: str
    result: Any
    success: bool
    error_message: Optional[str] = None

class MCPClient:
    """
    Single MCP client that efficiently handles database operations.
    Uses connection pooling and proper error handling.
    """
    
    def __init__(self, server_script_path: str, database_path: str):
        self.server_script_path = server_script_path
        self.database_path = database_path
        self.available_tools: List[Dict[str, Any]] = []
        self._tools_loaded = False

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call via MCP server (synchronous wrapper)."""
        return asyncio.run(self._execute_tool_call_async(tool_name, arguments))

    async def _execute_tool_call_async(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool call via MCP server"""
        try:
            # Use sys.executable to ensure we use the same Python interpreter
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_script_path, self.database_path]
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                
                async with ClientSession(read_stream, write_stream) as client:
                    # Initialize the client
                    await client.initialize()
                    
                    # Execute the tool call
                    result = await client.call_tool(tool_name, arguments)
                    
                    # Extract the actual text content
                    try:
                        if hasattr(result, 'content') and result.content:
                            # Try to get the text from the first content item
                            if isinstance(result.content, list) and len(result.content) > 0:
                                first_content = result.content[0]
                                if hasattr(first_content, 'text'):
                                    content_text = first_content.text
                                elif hasattr(first_content, 'type') and first_content.type == 'text':
                                    content_text = str(first_content)
                                else:
                                    # If it's a structured result, extract the data
                                    content_text = str(first_content)
                            else:
                                content_text = str(result.content)
                        elif hasattr(result, 'text'):
                            content_text = result.text
                        elif hasattr(result, 'data'):
                            content_text = json.dumps(result.data)
                        else:
                            # Last resort - convert to string but log for debugging
                            content_text = str(result)
                    except Exception as e:
                        content_text = str(result)
                    
                    return ToolResult(tool_name=tool_name, result=content_text, success=True)
                    
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                result=None,
                success=False,
                error_message=str(e)
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server (synchronous wrapper)."""
        return asyncio.run(self._get_available_tools_async())

    async def _get_available_tools_async(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server"""
        if self._tools_loaded:
            return self.available_tools
            
        try:
            # Use sys.executable to ensure we use the same Python interpreter
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_script_path, self.database_path]
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                
                async with ClientSession(read_stream, write_stream) as client:
                    await client.initialize()
                    tools_response = await client.list_tools()
                    self.available_tools = [tool.model_dump() for tool in tools_response.tools]
                    self._tools_loaded = True
                    
                    # Loaded MCP tools successfully
                    return self.available_tools
        except Exception:
            return []

    def query_database(self, query: str, params: Optional[List] = None, limit: int = 100) -> ToolResult:
        """Convenience method for database queries"""
        arguments = {
            "query": query,
            "limit": limit
        }
        if params:
            arguments["params"] = params
            
        return self.execute_tool_call("query_database", arguments)

    def get_table_schema(self, table_name: str) -> ToolResult:
        """Get schema for a specific table"""
        return self.execute_tool_call("get_table_schema", {"table_name": table_name})

    def list_tables(self, include_system_tables: bool = False) -> ToolResult:
        """List all tables in the database"""
        return self.execute_tool_call("list_tables", {"include_system_tables": include_system_tables})

    def describe_database(self) -> ToolResult:
        """Get comprehensive database overview"""
        return self.execute_tool_call("describe_database", {})

    def search_tables(self, search_term: str, search_type: str = "both") -> ToolResult:
        """Search for tables containing specific terms"""
        return self.execute_tool_call("search_tables", {
            "search_term": search_term,
            "search_type": search_type
        })

    def test_connection(self) -> bool:
        """Test if MCP server connection is working"""
        try:
            tools = self.get_available_tools()
            return len(tools) > 0
        except Exception:
            return False

    def get_tools_description(self) -> str:
        """Get a formatted description of available tools"""
        if not self._tools_loaded:
            return "Tools not loaded yet. Call get_available_tools() first."
        
        if not self.available_tools:
            return "No tools available."
        
        description = "Available MCP Database Tools:\n\n"
        for tool in self.available_tools:
            description += f"**{tool['name']}**\n"
            description += f"   Description: {tool['description']}\n"
            description += f"   Parameters: {list(tool['inputSchema']['properties'].keys())}\n\n"
        
        return description

# Convenience function for backwards compatibility
def create_mcp_client(server_script_path: str, database_path: str) -> MCPClient:
    """Create and initialize MCP client"""
    client = MCPClient(server_script_path, database_path)
    # Pre-load available tools
    client.get_available_tools()
    return client
