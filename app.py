#!/usr/bin/env python3

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add agents directory to Python path
sys.path.append(os.path.join(str(Path(__file__).parent), "agents", "chat"))
from module import FlightBookingChatAgent

async def main():
    """Main application entry point for Flight Booking Chat Assistant"""
    
    print("🚀 Starting Flight Booking Chat Assistant...")
    print("=" * 60)
    
    try:
        # Load environment variables from .env file
        project_root = Path(__file__).parent
        env_path = project_root / ".env"
        load_dotenv(env_path)
        
        # Get project paths
        config_path = os.path.join(project_root, "agents", "chat", "config.yaml")
        server_path = os.path.join(project_root, "mcp-server", "database_server.py")
        database_path = os.path.join(project_root, "database", "flights.db")
        
        # Verify required files exist
        missing_files = []
        if not os.path.exists(config_path):
            missing_files.append(f"Config file: {config_path}")
        if not os.path.exists(server_path):
            missing_files.append(f"MCP server: {server_path}")
        if not os.path.exists(database_path):
            missing_files.append(f"Database: {database_path}")
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            sys.exit(1)
        
        # Verify environment variables
        if not os.environ.get('DEEPINFRA_API_KEY'):
            print("❌ DEEPINFRA_API_KEY environment variable not set")
            print("   Please set your DeepInfra API key in .env file")
            sys.exit(1)
        
        print("✅ DeepInfra API key loaded from .env file")
        
        print("✅ All required files found")
        print("✅ Environment variables configured")
        print("🔌 Initializing chat agent with MCP integration...")
        
        # Create chat agent
        agent = FlightBookingChatAgent(
            config_path=str(config_path), 
            db_path=str(database_path), 
            server_path=str(server_path)
        )
        
        # Initialize and test MCP connection
        print("🔧 Testing MCP client/server connection...")
        await agent.initialize()
        
        print("✅ Chat agent initialized successfully!")
        print("✅ MCP client/server connection verified!")
        print("🔒 Database access through MCP client/server only")
        print("=" * 60)
        
        # Start interactive chat loop
        await agent.chat_loop()
        
    except KeyboardInterrupt:
        print("\n👋 Chat session ended by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_info():
    """Show application information"""
    print("\n📋 Flight Booking Chat Assistant")
    print("   - Uses DeepInfra API with Llama model")
    print("   - Database access ONLY through MCP client/server")
    print("   - Supports flight search and booking")
    print("   - Read-only database operations for safety")

if __name__ == "__main__":
    show_info()
    asyncio.run(main())