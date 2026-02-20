# Healthcare Concierge Agent

A multi-turn conversational AI agent built with [BeeAI Framework](https://github.com/i-am-bee/bee-agent-framework) and [AgentStack SDK](https://github.com/AgentStack-AI/AgentStack) that helps users find information about healthcare, hospitals, and doctors.

## Overview

The Healthcare Concierge Agent acts as an intelligent router that:
- Understands user healthcare queries
- Plans responses using the Think tool
- Hands off to specialized agents for detailed information:
  - **HealthcareResearchAgent**: General disease, symptom, and treatment information
  - **HospitalsAgent**: Hospital search and information
  - **DoctorFinderAgent**: Doctor search and specialization queries

## Features

- **Multi-turn Conversations**: Maintains context across multiple interactions
- **Intelligent Routing**: Uses conditional requirements to ensure thoughtful responses
- **Real-time Streaming**: Streams responses and trajectory updates to the UI
- **LLM Flexibility**: Supports Google Gemini models via extension system
- **Session Memory**: Persists conversation history per user session

## Architecture

```
User Query → Concierge Agent → Think Tool → Handoff to Specialist
                    ↓
            Trajectory Updates
                    ↓
            Streamed Response
```

### Key Components

- **BeeAI RequirementAgent**: Core agent with conditional tool execution
- **UnconstrainedMemory**: Session-based conversation memory
- **HandoffTool**: Enables delegation to specialist agents
- **ThinkTool**: Forces planning before action
- **Trajectory Extension**: Real-time UI updates

## Prerequisites

- Python 3.13+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- UV package manager ([Install UV](https://docs.astral.sh/uv/))

## Installation

1. **Clone the repository**
```bash
cd A2A/healthcare-assistant/concierge-agent
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

3. **Install dependencies**
```bash
uv sync
```

## Usage

### Local Development

```bash
uv run concierge_agent
```

The agent will start on `http://localhost:8000`

### Docker Deployment

```bash
docker build -t healthcare-concierge .
docker run -p 8000:8000 --env-file .env healthcare-concierge
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `PRODUCTION_MODE` | Enable production mode | No (default: false) |

### Agent Configuration

Edit `concierge_agent/agentstack_concierge.py` to customize:

- **Instructions**: Modify the `instructions` list
- **Max Iterations**: Change `max_iterations` in `AgentExecutionConfig`
- **Tool Requirements**: Adjust `ConditionalRequirement` rules
- **LLM Parameters**: Update `ChatModelParameters` (temperature, max_tokens)

## How It Works

### 1. Session Memory Management

```python
memory = get_memory(context)  # Get or create session memory
history = [message async for message in context.load_history() ...]
await memory.add_many([to_framework_message(msg) for msg in history])
```

Maintains conversation context by loading historical messages into BeeAI's memory system.

### 2. Conditional Tool Execution

```python
requirements=[
    ConditionalRequirement(think_tool, force_at_step=1, ...),
    ConditionalRequirement(research_handoff_tool, max_invocations=1, ...),
]
```

- **Step 1**: Always uses ThinkTool to plan
- **Later steps**: Can handoff to specialists (max 1 time each)

### 3. Streaming Response

```python
async for event, meta in agent.run(user_prompt, ...):
    if meta.name == "final_answer":
        yield event.delta  # Stream to user
    elif meta.name == "success":
        yield trajectory.trajectory_metadata(...)  # Update UI
```

Provides real-time feedback on agent reasoning and tool execution.

## Example Interactions

**User**: "What are the best hospitals in New York?"
```
1. Think: "User wants hospital information in NYC"
2. Handoff: HospitalsAgent
3. Response: [Hospital search results]
```

**User**: "What causes diabetes?"
```
1. Think: "User wants disease information"
2. Handoff: HealthcareResearchAgent
3. Response: [Disease information]
```

## Development

### Project Structure

```
concierge-agent/
├── concierge_agent/
│   └── agentstack_concierge.py  # Main agent implementation
├── pyproject.toml               # Dependencies
├── Dockerfile                   # Container configuration
├── .env.example                 # Environment template
└── README.md                    # This file
```

### Dependencies

- `beeai-framework[a2a]==0.1.74`: Core agent framework
- `agentstack-sdk==0.4.3`: AgentStack integration
- `agent-framework-a2a==1.0.0b251120`: A2A protocol support

## Troubleshooting

### LLM Extension Missing
**Error**: "LLM extension missing"
**Solution**: Ensure the AgentStack platform provides LLM fulfillment

### Handoff Agent Not Found
**Error**: Agent not in `handoff_agents`
**Solution**: Verify specialist agents are deployed to AgentStack platform

### Memory Issues
**Problem**: Agent doesn't remember previous messages
**Solution**: Check `context.load_history()` is returning messages

## Contributing

Contributions welcome! Key areas:
- Additional specialist agents
- Enhanced error handling
- Performance optimizations
- Tool improvements

## License

[Add your license here]

## Contact

**Maintainer**: Souptik Maiti  
**GitHub**: https://github.com/souptikmaiti

## Related Agents

- [HealthcareResearchAgent](../research-agent/)
- [HospitalsAgent](../hospitals-agent/)
- [DoctorFinderAgent](../doctor-finder-agent/)