# PrivateProcureAI architecture

## High-level system flow

```text
User
 ↓
Streamlit UI
Receives the user's natural-language goal
 ↓
Intent Extraction
Converts the goal into structured requirements
 ↓
Planner
Breaks the goal into smaller tasks
 ↓
Specialized Agents
Different agents solve different tasks
 ↓
RAG
Retrieves relevant information from the knowledge base
 ↓
Constraint Validator
Checks hard rules such as budget
 ↓
Evaluation Agent
Reviews the proposed plan
 ↓
PASS → Final plan
FAIL → Send back for re-planning