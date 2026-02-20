import os
import json
from typing import Annotated
from dotenv import load_dotenv

# imports from the beeai_framework
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.message import AssistantMessage, UserMessage, AnyMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.adapters.agentstack.agents import AgentStackAgent
from beeai_framework.adapters.agentstack.agents.types import AgentStackAgentStatus

# imports from the agentstack_sdk
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.a2a.extensions.ui.agent_detail import EnvVar
from agentstack_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailContributor,
    AgentDetailTool,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)

from a2a.types import Message, Role, AgentSkill
from a2a.utils.message import get_message_text


server = Server()
memories: dict[str, UnconstrainedMemory] = {}

def get_memory(context: RunContext) -> UnconstrainedMemory:
    # Keep a per-session memory object so that the conversation stays stateful
    """Get or create session memory keyed by context_id"""
    context_id = getattr(context, "context_id", getattr(context, "session_id", "default"))
    if context_id not in memories:
        memories[context.context_id] = UnconstrainedMemory()
    return memories[context.context_id]

def to_framework_message(message: Message) -> AnyMessage:
    """Convert A2A Message to Agent BeeAI Framework Message"""
    match message.role:
        case Role.user:
            return UserMessage(get_message_text(message))
        case Role.agent:
            return AssistantMessage(get_message_text(message))
        case _:
            return AssistantMessage(get_message_text(message))
        
def summarize_for_trajectory(data: object, limit: int = 400) -> str:
    """
    Convert tool inputs/outputs to a readable, bounded string for trajectory updates.
    """
    try:
        text = data if isinstance(data, str) else json.dumps(data, default=str)
    except Exception:
        text = str(data)

    return text if len(text) <= limit else f"{text[:limit]}... [truncated]"

@server.agent(
    name="HealthcareConciergeAgent",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Welcome to the Healthcare Concierge Agent! I can help you find information about healthcare, hospitals, and doctors.",
        input_placeholder="Ask me about healthcare or hospitals or doctors...",
        programming_language="Python",
        framework="BeeAI",
        contributors=[
            AgentDetailContributor(name="Souptik Maiti", email="https://github.com/souptikmaiti"),
        ],
        variables=[
            EnvVar(
                name="GOOGLE_API_KEY",
                description="Google API Key",
                required=True
            )
        ]
    ),
    skills=[
        AgentSkill(
            id="HealthcareConciergeAgent",
            name="Healthcare Concierge Agent",
            description="I can help you find information about healthcare, hospitals, and doctors.",
            tags=["healthcare", "hospitals", "doctors"],
            examples=["What are the best hospitals in New York?", "Which doctors specialize in oncology?", "What is the cost of a routine checkup?"]
        )
    ],
    tools=[
        AgentDetailTool(
            name="Think",
            description="Plans the best approach before responding."
        ),
        AgentDetailTool(
            name="Handoff",
            description="Handoff to a specialist agent."
        )
    ]
)
async def healthcare_concierge_wrapper(
    message: Message, 
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-3-flash-preview",)),
    ]
):
    """Healthcare Concierge Agent Wrapper for AgentStack"""

    # Initial UI update to the trajecotry so users see the agent is starting
    yield trajectory.trajectory_metadata(
        title="Initializing Agent...",
        content="Setting up your Healthcare Concierge.",
    )

    # Attach existing session memory and backfill prior messages for multi-turn conversation
    memory = get_memory(context)

    # Load existing history into BeeAI memory
    # context.load_history(): Agent Stack's A2A protocol history (platform-level)-> Persists messages across agent restarts -> Shared across the A2A ecosystem
    history = [message async for message in context.load_history() if isinstance(message, Message) and message.parts]
    
    # UnconstrainedMemory: BeeAI Framework's internal memory (agent-level) -> Required by RequirementAgent to function
    # Stores messages in BeeAI's format (AssistantMessage, UserMessage) -> Needed for the agent's reasoning and tool execution
    await memory.add_many([to_framework_message(msg) for msg in history])

    # Configure LLM from extension fulfillment
    if not llm or not llm.data:
        yield trajectory.trajectory_metadata(title="LLM Error", content="LLM extension missing.")
        yield "LLM selection is required."
        return

    llm_config = llm.data.llm_fulfillments.get("default")
    if not llm_config:
        yield trajectory.trajectory_metadata(title="LLM Error", content="No LLM fulfillment available.")
        yield "No LLM configuration available from the extension."
        return
    
    llm_client = GeminiChatModel(
        model_id=llm_config.api_model,
        api_key=llm_config.api_key,
        parameters=ChatModelParameters(
            temperature=0,
            max_tokens=4096,
            stream=True,
        ),
        tool_choice_support={"auto", "required"},
        allow_parallel_tool_calls=True,
    )

    # Make other AgentStack agents discoverable that have been deployed to the platform and make them available via handoff tools
    agentstack_agents = await AgentStackAgent.from_agent_stack()

    handoff_agents = {a.name: a for a in agentstack_agents if a.name in ("HealthcareResearchAgent", "HospitalsAgent", "DoctorFinderAgent")}

    research_handoff_tool = HandoffTool(handoff_agents["HealthcareResearchAgent"])
    hospitals_handoff_tool = HandoffTool(handoff_agents["HospitalsAgent"])
    doctors_handoff_tool = HandoffTool(handoff_agents["DoctorFinderAgent"])

    think_tool=ThinkTool()

    instructions = [
        "You are a helpful assistant that helps users find information about healthcare, hospitals, and doctors.",
        "You can handoff to the HealthcareResearchAgent if the user asks for general information about diseases, symptoms, causes, and treatments.",
        "You can handoff to the HospitalsAgent if the user asks for information about hospitals.",
        "You can handoff to the DoctorFinderAgent if the user asks for information about doctors.",
        "You can use the think tool to plan your response.",
        "If unsure, ask clarifying questions before giving guidance."
    ]

    # BeeAI RequirementAgent with Think and Handoff tools and Conditional Requirements
    agent = RequirementAgent(
        llm=llm_client,
        memory=memory,
        tools=[
            think_tool,
            research_handoff_tool,
            hospitals_handoff_tool,
            doctors_handoff_tool,
        ],
        instructions=instructions,
        role="Healthcare Concierge to handover to other agents",
        requirements=[
            ConditionalRequirement(think_tool, force_at_step=1, force_after=Tool, consecutive_allowed=False),
            ConditionalRequirement(research_handoff_tool, max_invocations=1, consecutive_allowed=False),
            ConditionalRequirement(hospitals_handoff_tool, max_invocations=1, consecutive_allowed=False),
            ConditionalRequirement(doctors_handoff_tool, max_invocations=1, consecutive_allowed=False),
        ],
    )

    user_prompt = get_message_text(message)

    response_text = ""

    def handle_final_answer_stream(data, meta) -> None:
        nonlocal response_text
        # Accumulate streamed final answer text
        if getattr(data, "delta", None):
            response_text += data.delta
            

    # Run the agent loop and stream trajectory and final answer
    async for event, meta in agent.run(
        user_prompt,
        execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2)
    ).on("final_answer", handle_final_answer_stream):
        if meta.name == "final_answer":
            if getattr(event, "delta", None):
                yield event.delta
            elif getattr(event, "text", None):
                response_text += event.text
        elif meta.name == "success" and event.state.steps:
            step = event.state.steps[-1]
            if step.tool and step.tool.name == "think":
                thoughts = step.input.get("thoughts", "Planning response...")
                yield trajectory.trajectory_metadata(title="Thinking...", content=thoughts)
            elif step.tool:
                tool_name = step.tool.name
                if tool_name != "final_answer":
                    yield trajectory.trajectory_metadata(title=f"{tool_name} (request)", content=summarize_for_trajectory(step.input))

                    if getattr(step, "error", None):
                        yield trajectory.trajectory_metadata(title=f"{tool_name} (error)", content=step.error.explain())
                    else:
                        output_text = step.output.get_text_content() if getattr(step, "output", None) else "No output"
                        yield trajectory.trajectory_metadata(title=f"{tool_name} (response)", content=summarize_for_trajectory(output_text))

    # Persist the final response in conversation history
    await context.store(AgentMessage(text=response_text))

def run():
    """Run the server"""
    server.run(
        host="0.0.0.0",
        port=8000,
        log_level="debug"
    )

