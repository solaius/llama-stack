# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import shutil
import tempfile
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Union

from llama_stack.apis.agents import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    Agents,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResumeRequest,
    Document,
    ListAgentSessionsResponse,
    ListAgentsResponse,
    Session,
    Turn,
)
from llama_stack.providers.inline.agents.meta_reference.persistence import AgentSessionInfo
from llama_stack.apis.inference import (
    Inference,
    ToolConfig,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl, kvstore_impl

from .agent_instance import ChatAgent
from .config import MetaReferenceAgentsImplConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MetaReferenceAgentsImpl(Agents):
    def __init__(
        self,
        config: MetaReferenceAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api

        self.in_memory_store = InmemoryKVStoreImpl()
        self.tempdir = tempfile.mkdtemp()
        
        # Keep track of agent IDs for the list endpoint
        self.agent_ids = []

    async def initialize(self) -> None:
        self.persistence_store = await kvstore_impl(self.config.persistence_store)

        # check if "bwrap" is available
        if not shutil.which("bwrap"):
            logger.warning("Warning: `bwrap` is not available. Code interpreter tool will not work correctly.")
            
        # Load the agent IDs from the persistence store
        agent_ids_json = await self.persistence_store.get("agent_ids")
        if agent_ids_json:
            try:
                self.agent_ids = json.loads(agent_ids_json)
            except Exception as e:
                logger.error(f"Error loading agent IDs: {str(e)}")
                self.agent_ids = []

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        agent_id = str(uuid.uuid4())
        created_at = datetime.now()

        await self.persistence_store.set(
            key=f"agent:{agent_id}",
            value=agent_config.model_dump_json(),
        )
        
        # Store the creation time
        await self.persistence_store.set(
            key=f"agent:{agent_id}:created_at",
            value=created_at.isoformat(),
        )
        
        # Track the agent ID for the list endpoint
        self.agent_ids.append(agent_id)

        # Also store a list of all agent IDs for persistence
        await self.persistence_store.set(
            key="agent_ids",
            value=json.dumps(self.agent_ids),
        )
        
        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def _get_agent_impl(self, agent_id: str) -> ChatAgent:
        agent_config = await self.persistence_store.get(
            key=f"agent:{agent_id}",
        )
        if not agent_config:
            raise ValueError(f"Could not find agent config for {agent_id}")

        try:
            agent_config = json.loads(agent_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not JSON decode agent config for {agent_id}") from e

        try:
            agent_config = AgentConfig(**agent_config)
        except Exception as e:
            raise ValueError(f"Could not validate(?) agent config for {agent_id}") from e

        return ChatAgent(
            agent_id=agent_id,
            agent_config=agent_config,
            tempdir=self.tempdir,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            vector_io_api=self.vector_io_api,
            tool_runtime_api=self.tool_runtime_api,
            tool_groups_api=self.tool_groups_api,
            persistence_store=(
                self.persistence_store if agent_config.enable_session_persistence else self.in_memory_store
            ),
        )

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        agent = await self._get_agent_impl(agent_id)

        session_id = await agent.create_session(session_name)
        return AgentSessionCreateResponse(
            session_id=session_id,
        )

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: List[
            Union[
                UserMessage,
                ToolResponseMessage,
            ]
        ],
        toolgroups: Optional[List[AgentToolGroup]] = None,
        documents: Optional[List[Document]] = None,
        stream: Optional[bool] = False,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )
        if stream:
            return self._create_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _create_agent_turn_streaming(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.create_and_execute_turn(request):
            yield event

    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: List[ToolResponse],
        stream: Optional[bool] = False,
    ) -> AsyncGenerator:
        request = AgentTurnResumeRequest(
            agent_id=agent_id,
            session_id=session_id,
            turn_id=turn_id,
            tool_responses=tool_responses,
            stream=stream,
        )
        if stream:
            return self._continue_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _continue_agent_turn_streaming(
        self,
        request: AgentTurnResumeRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.resume_turn(request):
            yield event

    async def get_agents_turn(self, agent_id: str, session_id: str, turn_id: str) -> Turn:
        agent = await self._get_agent_impl(agent_id)
        turn = await agent.storage.get_session_turn(session_id, turn_id)
        return turn

    async def get_agents_step(self, agent_id: str, session_id: str, turn_id: str, step_id: str) -> AgentStepResponse:
        turn = await self.get_agents_turn(agent_id, session_id, turn_id)
        for step in turn.steps:
            if step.step_id == step_id:
                return AgentStepResponse(step=step)
        raise ValueError(f"Provided step_id {step_id} could not be found")

    async def get_agents_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session:
        agent = await self._get_agent_impl(agent_id)
        session_info = await agent.storage.get_session_info(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found")
        turns = await agent.storage.get_session_turns(session_id)
        if turn_ids:
            turns = [turn for turn in turns if turn.turn_id in turn_ids]
        return Session(
            session_name=session_info.session_name,
            session_id=session_id,
            turns=turns,
            started_at=session_info.started_at,
        )

    async def delete_agents_session(self, agent_id: str, session_id: str) -> None:
        await self.persistence_store.delete(f"session:{agent_id}:{session_id}")

    async def delete_agent(self, agent_id: str) -> None:
        await self.persistence_store.delete(f"agent:{agent_id}")
        await self.persistence_store.delete(f"agent:{agent_id}:created_at")
        
        # Remove the agent ID from the list
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
            
            # Update the list in the persistence store
            await self.persistence_store.set(
                key="agent_ids",
                value=json.dumps(self.agent_ids),
            )

    async def shutdown(self) -> None:
        pass

    async def list_agents(self) -> ListAgentsResponse:
        """List all agents.

        :returns: A ListAgentsResponse with a list of available agents.
        """
        agents = []

        # Use the tracked agent IDs to get the agent configs
        for agent_id in self.agent_ids:
            try:
                agent_config_json = await self.persistence_store.get(f"agent:{agent_id}")
                if agent_config_json:
                    agent_config_dict = json.loads(agent_config_json)
                    agent_config = AgentConfig(**agent_config_dict)
                    
                    # Get the creation time if available
                    created_at_json = await self.persistence_store.get(f"agent:{agent_id}:created_at")
                    created_at = datetime.now()
                    if created_at_json:
                        try:
                            created_at = datetime.fromisoformat(created_at_json)
                        except Exception as e:
                            logger.error(f"Error parsing created_at for agent {agent_id}: {str(e)}")
                    
                    agents.append(
                        Agent(
                            agent_id=agent_id,
                            agent_config=agent_config,
                            created_at=created_at,
                        )
                    )
            except Exception as e:
                logger.error(f"Error loading agent {agent_id}: {str(e)}")

        # For in-memory store, also check for any agents that might not be in our tracked list
        if isinstance(self.persistence_store, InmemoryKVStoreImpl):
            for key, value in self.persistence_store._store.items():
                if key.startswith("agent:") and key != "agent_ids" and ":" not in key[6:]:
                    agent_id = key[6:]  # Remove "agent:" prefix

                    # Skip if we've already processed this agent
                    if agent_id in [agent.agent_id for agent in agents]:
                        continue

                    try:
                        agent_config_dict = json.loads(value)
                        agent_config = AgentConfig(**agent_config_dict)
                        
                        # Get the creation time if available
                        created_at_json = self.persistence_store._store.get(f"agent:{agent_id}:created_at")
                        created_at = datetime.now()
                        if created_at_json:
                            try:
                                created_at = datetime.fromisoformat(created_at_json)
                            except Exception as e:
                                logger.error(f"Error parsing created_at for agent {agent_id}: {str(e)}")
                        
                        agents.append(
                            Agent(
                                agent_id=agent_id,
                                agent_config=agent_config,
                                created_at=created_at,
                            )
                        )

                        # Add to our tracked list for future reference
                        if agent_id not in self.agent_ids:
                            self.agent_ids.append(agent_id)
                    except Exception as e:
                        logger.error(f"Error loading agent {agent_id}: {str(e)}")

            # Update the agent_ids in the persistence store
            await self.persistence_store.set(
                key="agent_ids",
                value=json.dumps(self.agent_ids),
            )

        return ListAgentsResponse(data=agents)

    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID.

        :param agent_id: ID of the agent.
        :returns: An Agent of the agent.
        """
        agent_config_json = await self.persistence_store.get(f"agent:{agent_id}")
        if not agent_config_json:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent_config_dict = json.loads(agent_config_json)
        agent_config = AgentConfig(**agent_config_dict)
        
        # Get the creation time if available
        created_at_json = await self.persistence_store.get(f"agent:{agent_id}:created_at")
        created_at = datetime.now()
        if created_at_json:
            try:
                created_at = datetime.fromisoformat(created_at_json)
            except Exception as e:
                logger.error(f"Error parsing created_at for agent {agent_id}: {str(e)}")
        
        return Agent(
            agent_id=agent_id,
            agent_config=agent_config,
            created_at=created_at,
        )

    async def list_agent_sessions(
        self,
        agent_id: str,
    ) -> ListAgentSessionsResponse:
        """List all session(s) of a given agent.

        :param agent_id: The ID of the agent to list sessions for.
        :returns: A ListAgentSessionsResponse with a list of sessions.
        """
        sessions = []
        
        # Check if agent exists
        agent_config_json = await self.persistence_store.get(f"agent:{agent_id}")
        if not agent_config_json:
            raise ValueError(f"Agent {agent_id} not found")
        
        # For in-memory store, we can iterate through the keys
        if isinstance(self.persistence_store, InmemoryKVStoreImpl):
            for key, value in self.persistence_store._store.items():
                if key.startswith(f"session:{agent_id}:") and key.count(":") == 2:
                    session_id = key.split(":")[-1]
                    
                    try:
                        session_info_dict = json.loads(value)
                        session_info = AgentSessionInfo(**session_info_dict)
                        
                        # Get the turns for this session
                        turns = []
                        for turn_key, turn_value in self.persistence_store._store.items():
                            if turn_key.startswith(f"session:{agent_id}:{session_id}:") and turn_key.count(":") == 3:
                                try:
                                    turn_dict = json.loads(turn_value)
                                    turn = Turn(**turn_dict)
                                    turns.append(turn)
                                except Exception as e:
                                    logger.error(f"Error loading turn {turn_key}: {str(e)}")
                        
                        sessions.append(
                            Session(
                                session_id=session_info.session_id,
                                session_name=session_info.session_name,
                                turns=turns,
                                started_at=session_info.started_at,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error loading session {key}: {str(e)}")
        else:
            # For other persistence stores, we need to list all keys and filter
            # This might not be efficient for large numbers of sessions
            # A better approach would be to add a list_keys method to the KVStore interface
            # that supports prefix filtering
            all_keys = await self.persistence_store.list_keys()
            for key in all_keys:
                if key.startswith(f"session:{agent_id}:") and key.count(":") == 2:
                    try:
                        value = await self.persistence_store.get(key)
                        if value:
                            session_info_dict = json.loads(value)
                            session_info = AgentSessionInfo(**session_info_dict)
                            
                            # Get the turns for this session
                            session_id = key.split(":")[-1]
                            turns = []
                            for turn_key in all_keys:
                                if turn_key.startswith(f"session:{agent_id}:{session_id}:") and turn_key.count(":") == 3:
                                    try:
                                        turn_value = await self.persistence_store.get(turn_key)
                                        if turn_value:
                                            turn_dict = json.loads(turn_value)
                                            turn = Turn(**turn_dict)
                                            turns.append(turn)
                                    except Exception as e:
                                        logger.error(f"Error loading turn {turn_key}: {str(e)}")
                            
                            sessions.append(
                                Session(
                                    session_id=session_info.session_id,
                                    session_name=session_info.session_name,
                                    turns=turns,
                                    started_at=session_info.started_at,
                                )
                            )
                    except Exception as e:
                        logger.error(f"Error loading session {key}: {str(e)}")
        
        return ListAgentSessionsResponse(data=sessions)
