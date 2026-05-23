#!/usr/bin/env python3
"""Refresh letta_v1 base_instructions and recompile agent + conversation system messages."""

import argparse
import asyncio
import logging

from sqlalchemy import select

from letta.orm.agent import Agent as AgentModel
from letta.orm.conversation_messages import ConversationMessage as ConversationMessageModel
from letta.orm.message import Message as MessageModel
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType, UpdateAgent
from letta.schemas.message import MessageUpdate
from letta.server.db import db_registry
from letta.server.server import SyncServer
from letta.services.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

NEW_PROMPT_MARKER = "three-layer memory system"
OLD_PROMPT_MARKER = "Open and view files"


def _compiled_has_new_prompt(content) -> bool:
    text = content if isinstance(content, str) else str(content)
    return NEW_PROMPT_MARKER in text


async def _get_conversation_system_message_id(
    actor,
    conversation_id: str,
    *,
    include_deleted_pivot: bool,
) -> str | None:
    async with db_registry.async_session() as session:
        conditions = [
            ConversationMessageModel.conversation_id == conversation_id,
            ConversationMessageModel.organization_id == actor.organization_id,
            ConversationMessageModel.in_context == True,
            MessageModel.role == "system",
        ]
        if not include_deleted_pivot:
            conditions.append(ConversationMessageModel.is_deleted == False)
        query = (
            select(MessageModel.id)
            .join(ConversationMessageModel, MessageModel.id == ConversationMessageModel.message_id)
            .where(*conditions)
            .order_by(ConversationMessageModel.position)
            .limit(1)
        )
        return (await session.execute(query)).scalar_one_or_none()


async def recompile_conversation(
    conv_mgr: ConversationManager,
    conversation_id: str,
    actor,
) -> bool:
    try:
        compiled_content = await conv_mgr.recompile_system_message_for_conversation(
            conversation_id=conversation_id,
            actor=actor,
            update_timestamp=False,
            dry_run=False,
        )
    except ValueError:
        return False
    return _compiled_has_new_prompt(compiled_content)


async def refresh_agent(
    server: SyncServer,
    actor,
    conv_mgr: ConversationManager,
    agent_id: str,
    *,
    dry_run: bool,
    force_system: bool,
    list_conversations_fn,
    include_deleted_pivot: bool,
) -> tuple[int, int, bool]:
    agent = await server.agent_manager.get_agent_by_id_async(agent_id=agent_id, actor=actor)
    needs_system_update = force_system or NEW_PROMPT_MARKER not in (agent.system or "")
    if OLD_PROMPT_MARKER in (agent.system or ""):
        needs_system_update = True

    if needs_system_update:
        if dry_run:
            logger.info("would update agents.system for %s (%s)", agent_id, agent.name)
        else:
            new_system = gpt_system.get_system_text("letta_v1").strip()
            await server.update_agent_async(agent_id=agent_id, request=UpdateAgent(system=new_system), actor=actor)

    if not dry_run:
        await server.agent_manager.rebuild_system_prompt_async(
            agent_id=agent_id,
            actor=actor,
            force=True,
            update_timestamp=False,
        )

    conversations = await list_conversations_fn(agent_id)
    conv_updated = 0
    for conv in conversations:
        if dry_run:
            logger.info("would recompile conversation %s (%s)", conv.id, conv.summary or "")
            conv_updated += 1
            continue
        if await recompile_conversation(
            conv_mgr,
            conv.id,
            actor,
        ):
            conv_updated += 1
            logger.info("recompiled conversation %s (%s)", conv.id, conv.summary or conv.id)

    return len(conversations), conv_updated, needs_system_update


async def refresh_all(
    *,
    dry_run: bool,
    agent_id: str | None,
    force_system: bool,
    include_deleted: bool,
) -> None:
    server = SyncServer()
    actor = await server.user_manager.get_default_actor_async()
    conv_mgr = ConversationManager()

    async with db_registry.async_session() as session:
        query = select(AgentModel).where(
            AgentModel.is_deleted == False,
            AgentModel.agent_type == AgentType.letta_v1_agent.value,
        )
        if agent_id:
            query = query.where(AgentModel.id == agent_id)
        agents = (await session.execute(query)).scalars().all()

    from letta.orm.conversation import Conversation as ConversationModel

    async def list_agent_conversations(agent_id: str) -> list:
        async with db_registry.async_session() as session:
            conditions = [
                ConversationModel.organization_id == actor.organization_id,
                ConversationModel.agent_id == agent_id,
            ]
            if not include_deleted:
                conditions.append(ConversationModel.is_deleted == False)
            rows = (await session.execute(select(ConversationModel).where(*conditions))).scalars().all()
        return rows

    if not agents:
        logger.info("no letta_v1_agent rows matched")
        return

    agents_updated = 0
    convs_total = 0
    convs_updated = 0

    for row in agents:
        n_convs, n_updated, system_changed = await refresh_agent(
            server,
            actor,
            conv_mgr,
            row.id,
            dry_run=dry_run,
            force_system=force_system,
            list_conversations_fn=list_agent_conversations,
            include_deleted_pivot=include_deleted,
        )
        convs_total += n_convs
        convs_updated += n_updated
        if system_changed:
            agents_updated += 1
        logger.info(
            "agent %s (%s): system_refresh=%s conversations=%s/%s",
            row.id,
            row.name,
            system_changed,
            n_updated,
            n_convs,
        )

    logger.info(
        "refresh complete: agents=%s conversations=%s/%s dry_run=%s",
        agents_updated,
        convs_updated,
        convs_total,
        dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="Refresh letta_v1 system prompts for agents and conversations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent-id", help="Refresh a single agent only")
    parser.add_argument(
        "--force-system",
        action="store_true",
        help="Rewrite agents.system even if it already contains the new prompt marker",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Also recompile system messages for soft-deleted conversations",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        refresh_all(
            dry_run=args.dry_run,
            agent_id=args.agent_id,
            force_system=args.force_system,
            include_deleted=args.include_deleted,
        )
    )


if __name__ == "__main__":
    main()
