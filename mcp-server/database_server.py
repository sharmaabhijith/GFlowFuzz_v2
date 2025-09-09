#!/usr/bin/env python3

import asyncio
import json
import sqlite3
import sys
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ListToolsResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database-mcp-server")

class DatabaseMCPServer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.server = Server("database-server")
        self.setup_tools()
    
    def setup_tools(self):
        """Register all available read-only tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available read-only database tools"""
            return [
                Tool(
                    name="query_database",
                    description="Execute a SELECT query on the database. Only SELECT statements are allowed.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL SELECT query to execute (must start with SELECT)"
                            },
                            "params": {
                                "type": "array",
                                "description": "Optional parameters for the SQL query to prevent SQL injection",
                                "items": {"type": ["string", "number", "null"]}
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of rows to return (default: 100, max: 1000)",
                                "minimum": 1,
                                "maximum": 1000
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get detailed schema information for a specific table including columns, types, constraints, and indexes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to get schema information for"
                            }
                        },
                        "required": ["table_name"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all tables and views in the database with their row counts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_system_tables": {
                                "type": "boolean",
                                "description": "Whether to include SQLite system tables (default: false)"
                            }
                        }
                    }
                ),
                Tool(
                    name="describe_database",
                    description="Get a comprehensive overview of the database structure",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="search_tables",
                    description="Search for tables containing specific column names or table name patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "Term to search for in table names or column names"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["table_name", "column_name", "both"],
                                "description": "Where to search: table names, column names, or both (default: both)"
                            }
                        },
                        "required": ["search_term"]
                    }
                ),
                Tool(
                    name="get_foreign_keys",
                    description="Get foreign key relationships for a table or entire database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Specific table name (optional - if not provided, shows all foreign keys)"
                            }
                        }
                    }
                ),
                Tool(
                    name="explain_query",
                    description="Get the execution plan for a SELECT query without executing it",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SELECT query to explain"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """Handle tool calls - returns content list or structured data"""
            try:
                if name == "query_database":
                    return await self._handle_query_database(arguments)
                elif name == "get_table_schema":
                    return await self._handle_get_table_schema(arguments)
                elif name == "list_tables":
                    return await self._handle_list_tables(arguments)
                elif name == "describe_database":
                    return await self._handle_describe_database(arguments)
                elif name == "search_tables":
                    return await self._handle_search_tables(arguments)
                elif name == "get_foreign_keys":
                    return await self._handle_get_foreign_keys(arguments)
                elif name == "explain_query":
                    return await self._handle_explain_query(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Tool execution failed for {name}: {e}")
                # Return error as content list
                return [
                    TextContent(
                        type="text",
                        text=f"Error executing {name}: {str(e)}"
                    )
                ]

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get read-only database connection"""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _validate_select_query(self, query: str) -> None:
        """Validate that the query is a safe SELECT statement"""
        query_clean = query.strip().lower()
        
        # Must start with SELECT
        if not query_clean.startswith('select'):
            raise ValueError("Only SELECT queries are allowed")
        
        # Block potentially dangerous keywords
        dangerous_keywords = [
            'insert', 'update', 'delete', 'drop', 'create', 'alter',
            'pragma', 'attach', 'detach', 'vacuum', 'reindex'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_clean:
                raise ValueError(f"Query contains prohibited keyword: {keyword}")

    async def _handle_query_database(self, arguments: Dict[str, Any]):
        """Handle database queries"""
        query = arguments.get("query", "")
        params = arguments.get("params", [])
        limit = arguments.get("limit", 100)
        
        # Validate query
        self._validate_select_query(query)
        
        # Apply limit if not already present
        if "limit" not in query.lower():
            query += f" LIMIT {min(limit, 1000)}"
        
        conn = self._get_db_connection()
        try:
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            # Get column info
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            response_data = {
                "query": query,
                "parameters": params,
                "columns": column_names,
                "results": results,
                "row_count": len(results),
                "limited": len(results) == limit
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2, default=str)
                )
            ]
        finally:
            conn.close()

    async def _handle_get_table_schema(self, arguments: Dict[str, Any]):
        """Get detailed table schema information"""
        table_name = arguments.get("table_name")
        
        conn = self._get_db_connection()
        try:
            # Get table info
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            schema = [dict(row) for row in cursor.fetchall()]
            
            # Get foreign keys
            cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = [dict(row) for row in cursor.fetchall()]
            
            # Get indexes
            cursor = conn.execute(f"PRAGMA index_list({table_name})")
            indexes = [dict(row) for row in cursor.fetchall()]
            
            # Get index details
            index_details = {}
            for index in indexes:
                cursor = conn.execute(f"PRAGMA index_info({index['name']})")
                index_details[index['name']] = [dict(row) for row in cursor.fetchall()]
            
            # Get table creation SQL
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            create_sql = cursor.fetchone()
            
            response_data = {
                "table_name": table_name,
                "columns": schema,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "index_details": index_details,
                "create_sql": create_sql["sql"] if create_sql else None
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )
            ]
        finally:
            conn.close()

    async def _handle_list_tables(self, arguments: Dict[str, Any]):
        """List all tables with metadata"""
        include_system = arguments.get("include_system_tables", False)
        
        conn = self._get_db_connection()
        try:
            # Get tables and views
            if include_system:
                query = "SELECT name, type, sql FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
            else:
                query = "SELECT name, type, sql FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            
            cursor = conn.execute(query)
            tables = [dict(row) for row in cursor.fetchall()]
            
            # Get row counts for tables
            for table in tables:
                if table['type'] == 'table':
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table['name']}")
                        count_result = cursor.fetchone()
                        table['row_count'] = count_result['count']
                    except:
                        table['row_count'] = 'Error'
                else:
                    table['row_count'] = 'N/A (View)'
            
            response_data = {
                "tables": tables,
                "total_tables": len([t for t in tables if t['type'] == 'table']),
                "total_views": len([t for t in tables if t['type'] == 'view'])
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )
            ]
        finally:
            conn.close()

    async def _handle_describe_database(self, arguments: Dict[str, Any]):
        """Get comprehensive database overview"""
        conn = self._get_db_connection()
        try:
            # Get database info
            cursor = conn.execute("PRAGMA database_list")
            db_info = [dict(row) for row in cursor.fetchall()]
            
            # Get all tables and views
            cursor = conn.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
            objects = [dict(row) for row in cursor.fetchall()]
            
            # Get total rows across all tables
            total_rows = 0
            table_summary = []
            
            for obj in objects:
                if obj['type'] == 'table':
                    try:
                        # Row count
                        cursor = conn.execute(f"SELECT COUNT(*) as count FROM {obj['name']}")
                        count = cursor.fetchone()['count']
                        total_rows += count
                        
                        # Column count
                        cursor = conn.execute(f"PRAGMA table_info({obj['name']})")
                        columns = cursor.fetchall()
                        
                        table_summary.append({
                            "name": obj['name'],
                            "type": obj['type'],
                            "row_count": count,
                            "column_count": len(columns)
                        })
                    except:
                        table_summary.append({
                            "name": obj['name'],
                            "type": obj['type'],
                            "row_count": "Error",
                            "column_count": "Error"
                        })
                else:  # view
                    table_summary.append({
                        "name": obj['name'],
                        "type": obj['type'],
                        "row_count": "N/A",
                        "column_count": "N/A"
                    })
            
            response_data = {
                "database_info": db_info,
                "summary": {
                    "total_tables": len([o for o in objects if o['type'] == 'table']),
                    "total_views": len([o for o in objects if o['type'] == 'view']),
                    "total_rows": total_rows
                },
                "objects": table_summary
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )
            ]
        finally:
            conn.close()


    async def _handle_search_tables(self, arguments: Dict[str, Any]):
        """Search for tables and columns containing specific terms"""
        search_term = arguments.get("search_term", "").lower()
        search_type = arguments.get("search_type", "both")
        
        conn = self._get_db_connection()
        try:
            results = {
                "search_term": search_term,
                "search_type": search_type,
                "matching_tables": [],
                "matching_columns": []
            }
            
            # Get all tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row['name'] for row in cursor.fetchall()]
            
            # Search table names
            if search_type in ["table_name", "both"]:
                for table in tables:
                    if search_term in table.lower():
                        results["matching_tables"].append({
                            "table_name": table,
                            "match_type": "table_name"
                        })
            
            # Search column names
            if search_type in ["column_name", "both"]:
                for table in tables:
                    try:
                        cursor = conn.execute(f"PRAGMA table_info({table})")
                        columns = [dict(row) for row in cursor.fetchall()]
                        
                        for column in columns:
                            if search_term in column['name'].lower():
                                results["matching_columns"].append({
                                    "table_name": table,
                                    "column_name": column['name'],
                                    "column_type": column['type'],
                                    "match_type": "column_name"
                                })
                    except:
                        continue
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )
            ]
        finally:
            conn.close()

    async def _handle_get_foreign_keys(self, arguments: Dict[str, Any]):
        """Get foreign key relationships"""
        table_name = arguments.get("table_name")
        
        conn = self._get_db_connection()
        try:
            foreign_keys = []
            
            if table_name:
                # Get foreign keys for specific table
                cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
                fks = [dict(row) for row in cursor.fetchall()]
                for fk in fks:
                    fk['from_table'] = table_name
                    foreign_keys.append(fk)
            else:
                # Get all foreign keys in database
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row['name'] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor = conn.execute(f"PRAGMA foreign_key_list({table})")
                        fks = [dict(row) for row in cursor.fetchall()]
                        for fk in fks:
                            fk['from_table'] = table
                            foreign_keys.append(fk)
                    except:
                        continue
            
            response_data = {
                "table_name": table_name,
                "foreign_keys": foreign_keys,
                "total_foreign_keys": len(foreign_keys)
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )
            ]
        finally:
            conn.close()

    async def _handle_explain_query(self, arguments: Dict[str, Any]):
        """Get query execution plan"""
        query = arguments.get("query", "")
        
        # Validate query
        self._validate_select_query(query)
        
        conn = self._get_db_connection()
        try:
            # Get query plan
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = [dict(row) for row in cursor.fetchall()]
            
            response_data = {
                "query": query,
                "execution_plan": plan
            }
            
            # Return content list instead of CallToolResult
            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )
            ]
        finally:
            conn.close()

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python readonly_database_server.py <database-path>", file=sys.stderr)
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # Verify database exists and is readable
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.close()
        logger.info(f"Connected to read-only database: {db_path}")
    except Exception as e:
        logger.error(f"Failed to connect to database {db_path}: {e}")
        sys.exit(1)
    
    # Create and run server
    server = DatabaseMCPServer(db_path)
    logger.info("Starting Read-Only Database MCP Server...")
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())