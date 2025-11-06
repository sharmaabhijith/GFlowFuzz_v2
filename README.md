# Agents, Policies, and MCP Database

This document explains the system’s agents, the booking policies they follow, and how the Model Context Protocol (MCP) database integration works. It intentionally ignores training and dataset topics.

## Overview
- Agents orchestrate the conversation, simulate users, generate SQL, and verify bookings.
- Policies define mandatory and advisory rules for behavior and safety.
- MCP provides read‑only database tools used to query the SQLite flights database.

## Agents

### Chat Agent
- Path: `agents/chat/module.py`
- Config: `agents/chat/config.yaml`
- Role: Conversational booking assistant that manages context, extracts booking details, searches flights via SQL, and generates a concise final booking summary.
- Key integrations:
  - Uses `SQLCoderAgent` to convert user intent into safe SQL.
  - Calls the MCP database server (through `MCPClient`) to run read‑only queries.
  - Maintains booking context and produces a final summary for verification.

Minimal usage example:

```python
from agents.chat.module import FlightBookingChatAgent

agent = FlightBookingChatAgent(
    config_path="agents/chat/config.yaml",
    db_path="database/flights.db",
    server_path="mcp-server/database_server.py",
)

print(agent.generate_chat_message("Find me a morning flight from JFK to LHR in business next month"))
print("\nSummary:\n", agent.generate_booking_summary())
```

### User Agent
- Path: `agents/user/module.py`
- Config: `agents/user/config.yaml`
- Role: LLM‑backed user simulator that produces realistic user turns given a conversation history and an objective (used for auditing/testing flows).

Example:

```python
from agents.user.module import FlightBookingUserAgent

user = FlightBookingUserAgent("agents/user/config.yaml")
history = [{"role": "assistant", "content": "Where are you flying from and to?"}]
print(user.generate_user_message(history, "Fly NYC → London in 2026, 1 pax, business"))
```

### Coder Agent (SQL)
- Path: `agents/coder/module.py`
- Role: Generates validated, read‑only SQL from natural language + recent context.
- Method: `generate_sql_query(user_request, conversation_history=None)`
  - Returns `{ "sql_query": str, "success": bool, "error": Optional[str] }`
  - Validation enforces SELECT‑only and basic structure expectations before queries are executed.

### Verifier Agent
- Path: `agents/verifier/module.py`
- Config: `agents/verifier/config.yaml`
- Role: Validates the final booking summary against the database (exists or not) using the MCP client. Returns a result dict including `verification_complete`, `booking_exists`, and a simple reward signal.

Example (with an existing ChatAgent instance):

```python
from agents.verifier.module import BookingVerifierAgent
from mcp_client import create_mcp_client  # provided in mcp-client/mcp_client.py

mcp = create_mcp_client("mcp-server/database_server.py", "database/flights.db")
verifier = BookingVerifierAgent("agents/verifier/config.yaml")

summary = agent.generate_booking_summary()
print(verifier.verify_booking(summary, mcp))
```

## Policies

- Source file: `policies/booking_policies.yaml`
- Structure: Each policy has `id`, `title`, `category`, `policy_kind` (Mandatory or Advisory), `severity`, `description`, and `guidance`.
- Core themes reflected in system prompts (and expected agent behavior):
  - Confirmation before booking (E‑GEN‑3)
  - Banned destination handling (M‑REG‑1)
  - Seat availability sanity (M‑RES‑1)
  - Canonical USD pricing (M‑RES‑2)
  - Conceal internal tool details (M‑PRY‑2)
  - Advisory behaviors around clarification, precision, and profitability

The chat and user agent system prompts embed these policies to drive behavior. You can also load and apply them programmatically:

```python
import yaml

with open("policies/booking_policies.yaml", "r") as f:
    policy_bundle = yaml.safe_load(f)

policies = policy_bundle.get("policies", [])
mandatory = [p for p in policies if p.get("policy_kind") == "Mandatory"]
advisory = [p for p in policies if p.get("policy_kind") == "Advisory"]
```

## MCP Database

### What is MCP here?
The repository uses the Model Context Protocol (MCP) to expose a read‑only SQLite flight database as tools callable from agents. The client starts the server over stdio when needed (no long‑running service required).

### Components
- Server: `mcp-server/database_server.py`
  - Enforces SELECT‑only, blocks DDL/DML, adds a `LIMIT` when missing.
  - Tools exposed:
    - `query_database` — run SELECT with optional params and limit
    - `get_table_schema` — detailed table schema (columns, FKs, indexes, DDL)
    - `list_tables` — tables/views with row counts
    - `describe_database` — global overview and object summary
    - `search_tables` — search by table/column names
    - `get_foreign_keys` — FK relationships
    - `explain_query` — query plan without executing it
- Client: `mcp-client/mcp_client.py`
  - Class: `MCPClient(server_script_path, database_path)`
  - Convenience: `create_mcp_client(server_script_path, database_path)`
  - Returns `ToolResult` with `.success`, `.result` (JSON/text), `.error_message`.
- Database: `database/flights.db` (SQLite, read‑only access by server)

### Quick examples

Querying the database directly:

```python
import json, sys, os
sys.path.append("mcp-client")  # ensure module path for mcp_client.py
from mcp_client import MCPClient

client = MCPClient("mcp-server/database_server.py", "database/flights.db")

res = client.query_database("SELECT flight_number, price, currency FROM flights WHERE departure_airport='JFK' AND arrival_airport='LHR' ORDER BY price ASC")
if res.success:
    data = json.loads(res.result)
    print("Rows:", data.get("row_count"))
    print("First:", (data.get("results") or [None])[0])
else:
    print("Error:", res.error_message)
```

Exploring the schema:

```python
import json, sys
sys.path.append("mcp-client")
from mcp_client import MCPClient

client = MCPClient("mcp-server/database_server.py", "database/flights.db")

print("Tools:", [t["name"] for t in client.get_available_tools()])
schema = client.get_table_schema("flights")
print(json.loads(schema.result)["columns"])  # column definitions

tables = client.list_tables()
print(json.loads(tables.result)["tables"])  # tables and row counts
```

### How agents use MCP
1. Chat Agent asks `SQLCoderAgent` to generate a safe SELECT.
2. Chat Agent executes the SQL via `MCPClient.query_database(...)`.
3. Results are parsed and formatted for the user (with policy constraints like USD pricing and seat availability in the prompt).
4. Verifier Agent optionally checks the final booking summary with MCP to decide if it already exists.

## Environment
- API key: set `DEEPINFRA_API_KEY` in `.env` at the repo root.
- Default model and behavior are configured per agent under their `config.yaml` files.

## Notes
- This guide focuses on agents, policies, and the MCP database. It intentionally does not cover training pipelines or dataset preparation.

