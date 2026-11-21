"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int) -> dict:
    """
    Update WorkflowState using optimistic locking.

    The expected_version must be the version read before the worker call.
    If another update changes the version, reject this stale result.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    update_expr_parts = [f"{k} = :{k}" for k in updates]
    update_expr_parts.append("version = :new_version")
    update_expr = "SET " + ", ".join(update_expr_parts)

    expr_values = {f":{k}": v for k, v in updates.items()}
    expr_values[":new_version"] = expected_version + 1
    expr_values[":expected_version"] = expected_version

    try:
        table.update_item(
            Key={"session_id": session_id},
            UpdateExpression=update_expr,
            ConditionExpression="version = :expected_version",
            ExpressionAttributeValues=expr_values
        )
        return _read_workflow_state(session_id)

    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException as e:
        raise RuntimeError(
            f"WorkflowState version conflict for session {session_id}. "
            f"Expected version {expected_version}; worker result was not saved."
        ) from e

# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

# def build_inventory_agent() -> Agent:
#     """
#     Build the Inventory Agent.

#     Gathers order and customer facts from DynamoDB. Does NOT make decisions -
#     only retrieves data for the OrchestratorAgent to share with downstream agents.
#     """

#     # TODO: Create a BedrockModel using the WORKER model
#     pass

#     # TODO: System prompt for the Inventory Agent
#     pass

#     # TODO: Implement check_order_status
#     # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
#     # order_id = sort key), so a get_item needs BOTH values. That is why this
#     # tool takes customer_id as well as order_id.
#     @tool
#     def check_order_status(customer_id: str, order_id: str) -> dict:
#         """
#         Look up one order in DynamoDB and report its status, product, dates
#         and amount. Reports facts only - it does NOT decide return eligibility.

#         Args:
#             customer_id: The customer's unique identifier (e.g. CUST-001)
#             order_id: The order identifier (e.g. ORD-27176)

#         Returns:
#             Order record (order_id, status, product_name, order_date, price, ...)
#             or a not-found message
#         """
#         pass

#     # TODO: Implement get_customer_tier
#     @tool
#     def get_customer_tier(customer_id: str) -> dict:
#         """
#         Retrieve a customer's tier (Standard or Premium) from DynamoDB.
#         Standard customers have a 30-day return window; Premium customers have 60 days.

#         Args:
#             customer_id: The customer's unique identifier

#         Returns:
#             Customer profile including tier and account details
#         """
#         pass

#     # TODO: Implement list_customer_orders
#     @tool
#     def list_customer_orders(customer_id: str) -> dict:
#         """
#         Retrieve all orders for a customer from DynamoDB.

#         Args:
#             customer_id: The customer's unique identifier

#         Returns:
#             List of all orders with order_id, status, order_date, and amount
#         """
#         pass

#     # TODO: Instantiate and return the Agent
#     pass



def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """
    You are NovaMart's Inventory Agent.

    Your role is to retrieve accurate order and customer facts from DynamoDB.
    You must not make return eligibility decisions, interpret policies, or
    promise refunds. Report only facts supported by tool results.

    Use check_order_status when asked about a specific order.
    Use get_customer_tier for customer profile and tier information.
    Use list_customer_orders to retrieve a customer's orders.

    If a record is not found, clearly say so. Never invent order or customer data.
    """

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """Retrieve one order using its composite DynamoDB key."""
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                }
            )
            item = response.get("Item")
            if not item:
                return {
                    "found": False,
                    "message": (
                        f"No order {order_id} was found for customer {customer_id}."
                    ),
                }
            return {"found": True, "order": item}
        except ClientError as exc:
            logger.exception("Failed to retrieve order %s", order_id)
            return {
                "error": "Unable to retrieve the order from DynamoDB.",
                "details": str(exc),
            }

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """Retrieve a customer profile, including tier and contact preferences."""
        try:
            table = dynamodb.Table(config.CUSTOMERS_TABLE)
            response = table.get_item(Key={"customer_id": customer_id})
            item = response.get("Item")
            if not item:
                return {
                    "found": False,
                    "message": f"No customer {customer_id} was found.",
                }
            return {"found": True, "customer": item}
        except ClientError as exc:
            logger.exception("Failed to retrieve customer %s", customer_id)
            return {
                "error": "Unable to retrieve the customer profile from DynamoDB.",
                "details": str(exc),
            }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """Retrieve all orders belonging to a customer."""
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.query(
                KeyConditionExpression=Key("customer_id").eq(customer_id)
            )
            items = response.get("Items", [])

            while response.get("LastEvaluatedKey"):
                response = table.query(
                    KeyConditionExpression=Key("customer_id").eq(customer_id),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            return {
                "customer_id": customer_id,
                "count": len(items),
                "orders": items,
            }
        except ClientError as exc:
            logger.exception("Failed to list orders for %s", customer_id)
            return {
                "error": "Unable to list customer orders from DynamoDB.",
                "details": str(exc),
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

# def build_refund_agent() -> Agent:
#     """
#     Build the Refund Agent.

#     Makes return/refund eligibility decisions based on order facts from
#     WorkflowState and applies the correct policy window per customer tier.
#     """

#     # TODO: Create a BedrockModel
#     pass

#     # TODO: System prompt for the Refund Agent
#     pass

#     # TODO: Implement get_inventory_context
#     @tool
#     def get_inventory_context(session_id: str) -> dict:
#         """
#         Read the WorkflowState to access facts gathered by the InventoryAgent.

#         Args:
#             session_id: The current session identifier

#         Returns:
#             The inventory_agent field from WorkflowState, or empty dict if not yet set
#         """
#         pass

#     # TODO: Implement initiate_refund
#     @tool
#     def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
#         """
#         Initiate a return by updating the order record in DynamoDB.

#         Args:
#             customer_id: The customer's unique identifier
#             order_id: The order to return
#             reason: Customer-provided reason for the return

#         Returns:
#             Confirmation dict with return_reference number and instructions
#         """
#         pass

#     # TODO: Instantiate and return the Agent
#     pass


def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Evaluates return eligibility using verified DynamoDB order and
    customer data, then records approved return requests.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """
    You are NovaMart's Refund Agent.

    Your responsibilities:
    1. Read inventory findings using get_inventory_context.
    2. Evaluate return eligibility using verified order and customer facts.
    3. Apply the return windows: Standard customers have 30 days;
       Premium customers have 60 days.
    4. Only delivered orders within the applicable return window may
       be approved.
    5. Call initiate_refund only when the customer requests a return
       and the order is eligible.

    The initiate_refund tool independently validates eligibility and
    records the return request. Treat its result as authoritative.
    Never claim a return was approved unless the tool confirms success.
    Never invent a return reference or promise an unconfirmed refund.
    Explain denials clearly and professionally.
    """

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """Read inventory findings from the current WorkflowState."""
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return {
                    "found": False,
                    "message": (
                        f"No WorkflowState exists for session {session_id}."
                    ),
                }

            return {
                "found": True,
                "customer_id": state.get("customer_id"),
                "inventory_agent": state.get("inventory_agent", {}),
            }
        except ClientError as exc:
            logger.exception(
                "Unable to read inventory context for session %s",
                session_id,
            )
            return {
                "error": "Unable to read WorkflowState.",
                "details": str(exc),
            }

    @tool
    def initiate_refund(
        customer_id: str,
        order_id: str,
        reason: str,
    ) -> dict:
        """
        Validate return eligibility and atomically record a return request.

        Standard return window: 30 days.
        Premium return window: 60 days.
        Only delivered orders within the applicable window are eligible.
        """
        from datetime import datetime, timezone
        from botocore.exceptions import ClientError

        try:
            orders_table = dynamodb.Table(config.ORDERS_TABLE)
            customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

            # Verify the order directly in DynamoDB.
            order_response = orders_table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                }
            )
            order = order_response.get("Item")

            if not order:
                return {
                    "approved": False,
                    "message": (
                        f"Order {order_id} was not found for "
                        f"customer {customer_id}."
                    ),
                }

            # Verify the customer's actual tier.
            customer_response = customers_table.get_item(
                Key={"customer_id": customer_id}
            )
            customer = customer_response.get("Item")

            if not customer:
                return {
                    "approved": False,
                    "message": f"Customer {customer_id} was not found.",
                }

            tier = str(customer.get("tier", "")).strip().lower()

            if tier == "premium":
                return_window_days = 60
            elif tier == "standard":
                return_window_days = 30
            else:
                return {
                    "approved": False,
                    "message": "Customer tier could not be verified.",
                }

            # Only delivered orders can be returned.
            status = str(order.get("status", "")).strip().lower()

            if status != "delivered":
                return {
                    "approved": False,
                    "message": (
                        f"Order {order_id} has status '{status}'. "
                        "Only delivered orders are eligible for return."
                    ),
                }

            # Prevent duplicate return requests.
            if order.get("return_reference"):
                return {
                    "approved": False,
                    "message": (
                        f"Order {order_id} already has a return request "
                        f"({order['return_reference']})."
                    ),
                }

            # Validate the order date and applicable return window.
            order_date_text = order.get("order_date")

            if not order_date_text:
                return {
                    "approved": False,
                    "message": "The order date is missing; eligibility cannot be verified.",
                }

            try:
                order_date = datetime.strptime(
                    str(order_date_text)[:10], "%Y-%m-%d"
                ).date()
            except ValueError:
                return {
                    "approved": False,
                    "message": "The order date is invalid; eligibility cannot be verified.",
                }

            today = datetime.now(timezone.utc).date()
            age_days = (today - order_date).days

            if age_days < 0:
                return {
                    "approved": False,
                    "message": "The order date is in the future; eligibility cannot be verified.",
                }

            if age_days > return_window_days:
                return {
                    "approved": False,
                    "message": (
                        f"Order {order_id} is {age_days} days old. "
                        f"The {tier.title()} return window is "
                        f"{return_window_days} days, so this order is "
                        "not eligible for return."
                    ),
                }

            # Record the return only if the order is still delivered
            # and no other return request has been recorded.
            return_reference = (
                f"RET-{uuid.uuid4().hex[:10].upper()}"
            )
            requested_at = datetime.now(timezone.utc).isoformat()

            orders_table.update_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                },
                UpdateExpression=(
                    "SET return_status = :return_status, "
                    "return_reason = :reason, "
                    "return_reference = :reference, "
                    "return_requested_at = :requested_at"
                ),
                ConditionExpression=(
                    "attribute_exists(order_id) "
                    "AND #order_status = :delivered "
                    "AND attribute_not_exists(return_reference)"
                ),
                ExpressionAttributeNames={
                    "#order_status": "status",
                },
                ExpressionAttributeValues={
                    ":return_status": "requested",
                    ":reason": reason,
                    ":reference": return_reference,
                    ":requested_at": requested_at,
                    ":delivered": "delivered",
                },
            )

            return {
                "approved": True,
                "return_reference": return_reference,
                "customer_id": customer_id,
                "order_id": order_id,
                "return_status": "requested",
                "reason": reason,
                "order_age_days": age_days,
                "return_window_days": return_window_days,
                "message": (
                    "The return request has been recorded successfully. "
                    "This confirms the return request, not a completed refund."
                ),
            }

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")

            if error_code == "ConditionalCheckFailedException":
                return {
                    "approved": False,
                    "message": (
                        "The order is no longer eligible for this update, "
                        "or a return request already exists. Please recheck "
                        "the latest order status."
                    ),
                }

            logger.exception(
                "Failed to initiate return for order %s", order_id
            )
            return {
                "approved": False,
                "message": "Unable to record the return request due to a database error.",
                "details": str(exc),
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            get_inventory_context,
            initiate_refund,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent with three parallel policy retrievers.
    """

    # Returns policy retriever
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            config.RETURNS_KB_ID, query
        )
        return format_kb_results(results)

    returns_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=(
            "You are the NovaMart Returns Policy Retriever. "
            "Use retrieve_returns_policy to find relevant return-policy "
            "passages. Answer only from retrieved evidence. If no relevant "
            "evidence is found, state that clearly."
        ),
        tools=[retrieve_returns_policy],
    )

    # Shipping policy retriever
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            config.SHIPPING_KB_ID, query
        )
        return format_kb_results(results)

    shipping_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=(
            "You are the NovaMart Shipping Policy Retriever. "
            "Use retrieve_shipping_policy to find relevant shipping-policy "
            "passages. Answer only from retrieved evidence. If no relevant "
            "evidence is found, state that clearly."
        ),
        tools=[retrieve_shipping_policy],
    )

    # Warranty policy retriever
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            config.WARRANTY_KB_ID, query
        )
        return format_kb_results(results)

    warranty_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=(
            "You are the NovaMart Warranty Policy Retriever. "
            "Use retrieve_warranty_policy to find relevant warranty-policy "
            "passages. Answer only from retrieved evidence. If no relevant "
            "evidence is found, state that clearly."
        ),
        tools=[retrieve_warranty_policy],
    )

    # Query all three retrievers concurrently.
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query Returns, Shipping, and Warranty knowledge bases in parallel.
        Return the combined policy findings.
        """
        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }

        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """Run one specialist retriever and return its result."""
            try:
                response = agent(query)
                result_text = str(response)
                return domain, result_text
            except Exception as exc:
                logger.exception(
                    "The %s policy retriever failed", domain
                )
                return domain, f"Policy retrieval failed: {exc}"

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _run_retriever, domain, agent, query
                ): domain
                for domain, agent in retrievers.items()
            }

            for future in as_completed(futures):
                domain = futures[future]
                try:
                    result_domain, result_text = future.result()
                    results[result_domain] = result_text
                except Exception as exc:
                    logger.exception(
                        "Failed to collect %s policy results", domain
                    )
                    results[domain] = f"Policy retrieval failed: {exc}"

        trace.kb_done(len(retrievers))

        for domain in ["Returns", "Shipping", "Warranty"]:
            trace.kb_result(
                domain, results.get(domain, "[No results]")
            )

        return "\n\n".join(
            f"=== {domain} Policy ===\n"
            f"{results.get(domain, '[No results]')}"
            for domain in ["Returns", "Shipping", "Warranty"]
        )

    # Policy coordinator
    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    coordinator_prompt = """
    You are NovaMart's Policy Agent coordinator.

    Your task is to answer customer questions about returns, shipping,
    and warranties using retrieved policy evidence.

    Always call search_all_policies before answering a policy question.
    Ground your answer in the returned evidence. Do not invent policy
    terms, dates, eligibility rules, or guarantees.

    If the retrieved evidence is incomplete or does not answer the question,
    say what could not be verified. Clearly distinguish policy information
    from assumptions. Be concise and helpful.
    """

    return Agent(
        model=coordinator_model,
        system_prompt=coordinator_prompt,
        tools=[search_all_policies],
    )

# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────


def build_communication_agent() -> Agent:
    """Build the agent that drafts the final customer-facing response."""

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3,
    )

    system_prompt = """
    You are NovaMart's customer communication specialist.

    Read the complete WorkflowState using get_full_workflow_context before
    drafting the final response.

    Use the original customer request and the available agent findings.
    Be warm, professional, empathetic, concise, and factually accurate.
    Explain relevant order facts, policy findings, and return/refund outcomes
    only when those findings are present in the workflow state.

    Never invent an order status, policy, refund approval, refund reference,
    or action that has not occurred. If information is missing, explain that
    it could not be verified. Do not expose internal workflow details,
    implementation details, or raw system errors to the customer.

    For arithmetic questions, clearly show the calculation and the result.
    Do not round currency values until the full calculation is complete.
    """

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState for this customer session.
        """
        try:
            state = _read_workflow_state(session_id)
            if not state:
                return {
                    "found": False,
                    "session_id": session_id,
                    "message": "WorkflowState was not found.",
                }

            return {
                "found": True,
                "workflow_state": state,
            }
        except Exception:
            logger.exception(
                "Failed to read WorkflowState for session %s", session_id
            )
            return {
                "found": False,
                "session_id": session_id,
                "message": "Workflow context could not be retrieved.",
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent,
) -> Agent:
    """Build the Orchestrator that routes requests and manages WorkflowState."""

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
    )

    system_prompt = """
    You are NovaMart's request orchestrator. You coordinate specialist agents
    and must not write the final customer-facing response yourself.

    REQUIRED ORDER:
    1. Always call initialize_session first, before any other tool.
    2. Route the request according to the rules below.
    3. Always call route_to_communication_agent last.

    ROUTING RULES:
    - Order status, order history, delivery, or order facts:
      call route_to_inventory_agent first.
    - Return/refund requests involving an order:
      call route_to_inventory_agent first, then route_to_refund_agent.
    - Questions about return policy, shipping policy, or warranties:
      call route_to_policy_agent.
    - Account information or customer tier questions:
      call route_to_inventory_agent; do not call the Policy Agent just to
      answer account or tier questions.
    - Pure arithmetic or calculations:
      skip Inventory, Policy, and Refund; go directly to Communication.
    - For requests involving more than one category, call each relevant
      specialist in a sensible order, preserving inventory-before-refund.
    - Do not claim an action was completed unless the relevant worker result
      confirms it.
    - Do not answer the customer directly. Communication must be the final
      tool call for every request.

    Use the session_id and customer_id supplied by the customer context.
    Pass the original customer request to each relevant worker.
    """

    @tool
    def route_to_inventory_agent(
        session_id: str, customer_id: str, request: str
    ) -> str:
        """Gather order and customer facts from the Inventory Agent."""
        state = _read_workflow_state(session_id)
        if not state:
            return "WorkflowState not found. Initialize the session first."

        old_version = int(state.get("version", 0))
        trace.step_start("inventory_agent")

        try:
            response = inventory_agent(
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n"
                "Gather the relevant order/customer facts using your tools. "
                "Do not make refund eligibility decisions."
            )
            result = str(response)
            _update_workflow_state(
                session_id,
                {"inventory_agent": result},
                old_version,
            )
            trace.step_done("inventory_agent", old_version)
            return result
        except Exception as exc:
            logger.exception("Inventory Agent routing failed")
            return f"Inventory Agent could not complete the request: {exc}"

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """Retrieve policy information using the Policy Agent."""
        state = _read_workflow_state(session_id)
        if not state:
            return "WorkflowState not found. Initialize the session first."

        old_version = int(state.get("version", 0))
        trace.step_start("policy_agent")

        try:
            response = policy_agent(
                f"Session ID: {session_id}\n"
                f"Policy question: {request}"
            )
            result = str(response)
            _update_workflow_state(
                session_id,
                {"policy_agent": result},
                old_version,
            )
            trace.step_done("policy_agent", old_version)
            return result
        except Exception as exc:
            logger.exception("Policy Agent routing failed")
            return f"Policy information could not be retrieved: {exc}"

    @tool
    def route_to_refund_agent(
        session_id: str, customer_id: str, request: str
    ) -> str:
        """Process a return/refund request after inventory facts are gathered."""
        state = _read_workflow_state(session_id)
        if not state:
            return "WorkflowState not found. Initialize the session first."

        if not state.get("inventory_agent"):
            return (
                "Inventory facts are missing. Call "
                "route_to_inventory_agent before processing a return/refund."
            )

        old_version = int(state.get("version", 0))
        trace.step_start("refund_agent")

        try:
            response = refund_agent(
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n"
                "Use get_inventory_context to inspect the stored inventory "
                "findings. Verify eligibility before initiating a return."
            )
            result = str(response)
            _update_workflow_state(
                session_id,
                {"refund_agent": result},
                old_version,
            )
            trace.step_done("refund_agent", old_version)
            return result
        except Exception as exc:
            logger.exception("Refund Agent routing failed")
            return f"The return/refund request could not be processed: {exc}"

    @tool
    def route_to_communication_agent(
        session_id: str, customer_id: str, original_request: str
    ) -> str:
        """Draft the final customer-facing response. Always call this last."""
        state = _read_workflow_state(session_id)
        if not state:
            return "WorkflowState not found. Initialize the session first."

        old_version = int(state.get("version", 0))
        trace.step_start("communication_agent")

        try:
            response = communication_agent(
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Original customer request: {original_request}\n"
                "Read the full WorkflowState and draft the final response."
            )
            result = str(response)
            _update_workflow_state(
                session_id,
                {"communication_agent": result},
                old_version,
            )
            trace.step_done("communication_agent", old_version)
            return result
        except Exception as exc:
            logger.exception("Communication Agent routing failed")
            return (
                "I'm sorry, but I couldn't prepare a complete response "
                "right now. Please try again."
            )

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """Create or validate the WorkflowState for a customer session."""
        try:
            existing = _read_workflow_state(session_id)

            if existing:
                if existing.get("customer_id") != customer_id:
                    return (
                        "Session initialization failed: the session belongs "
                        "to a different customer."
                    )
                return (
                    f"Session {session_id} is already initialized for "
                    f"customer {customer_id}."
                )

            _create_workflow_state(session_id, customer_id)
            return (
                f"Initialized session {session_id} for customer {customer_id}."
            )
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # Another invocation may have created the session concurrently.
            existing = _read_workflow_state(session_id)
            if existing and existing.get("customer_id") == customer_id:
                return f"Session {session_id} is initialized."
            return "Session initialization failed."
        except Exception as exc:
            logger.exception("Session initialization failed")
            return f"Session initialization failed: {exc}"

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════


def create_guardrail() -> tuple[str, str]:
    """Create the project guardrail and return its ID and numbered version."""
    bedrock_client = boto3.client(
        "bedrock",
        region_name=config.AWS_REGION,
    )

    # Find an existing guardrail with the configured name.
    existing = bedrock_client.list_guardrails()
    while True:
        for guardrail in existing.get("guardrails", []):
            if guardrail.get("name") != config.GUARDRAIL_NAME:
                continue

            guardrail_id = guardrail["id"]
            listed_version = guardrail.get("version", "DRAFT")

            # Reuse an existing numbered version.
            if listed_version != "DRAFT":
                print(
                    f"Guardrail already exists: {guardrail_id} "
                    f"(version: {listed_version})"
                )
                return guardrail_id, listed_version

            # Confirm that the draft exists before versioning it.
            bedrock_client.get_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion="DRAFT",
            )

            version_response = bedrock_client.create_guardrail_version(
                guardrailIdentifier=guardrail_id,
                description="Versioned NovaMart project guardrail",
            )
            guardrail_version = version_response["version"]

            print(
                f"Created numbered guardrail version: "
                f"{guardrail_id} ({guardrail_version})"
            )
            return guardrail_id, guardrail_version

        next_token = existing.get("nextToken")
        if not next_token:
            break

        existing = bedrock_client.list_guardrails(
            nextToken=next_token,
        )

    # Content filters: stronger filters for sexual content, violence, and hate;
    # medium-strength filters for insults and misconduct.
    filters = [
        {
            "type": filter_type,
            "inputStrength": strength,
            "outputStrength": strength,
            "inputAction": "BLOCK",
            "outputAction": "BLOCK",
            "inputEnabled": True,
            "outputEnabled": True,
        }
        for filter_type, strength in [
            ("SEXUAL", "HIGH"),
            ("VIOLENCE", "HIGH"),
            ("HATE", "HIGH"),
            ("INSULTS", "MEDIUM"),
            ("MISCONDUCT", "MEDIUM"),
        ]
    ]

    # Block financial card numbers and US SSNs; anonymize emails and phones.
    pii_entities = [
        {
            "type": "CREDIT_DEBIT_CARD_NUMBER",
            "action": "BLOCK",
            "inputEnabled": True,
            "outputEnabled": True,
        },
        {
            "type": "US_SOCIAL_SECURITY_NUMBER",
            "action": "BLOCK",
            "inputEnabled": True,
            "outputEnabled": True,
        },
        {
            "type": "EMAIL",
            "action": "ANONYMIZE",
            "inputEnabled": True,
            "outputEnabled": True,
        },
        {
            "type": "PHONE",
            "action": "ANONYMIZE",
            "inputEnabled": True,
            "outputEnabled": True,
        },
    ]

    topic_definitions = {
        "competitor products": (
            "Requests to recommend, compare, promote, or provide product "
            "details specifically about competing retailers or their products. "
            "Do not block ordinary questions about the customer's NovaMart order."
        ),
        "pricing negotiations": (
            "Requests to bargain, haggle, negotiate, or change an advertised "
            "product price, including asking for a special price or an "
            "unauthorized discount. Allow arithmetic using a price and discount "
            "already specified by the user, including calculating the discounted "
            "price, savings, totals, and percentages. Do not treat calculation "
            "alone as price negotiation."
        ),
        "legal threats": (
            "Requests to draft, formulate, or escalate legal threats, including "
            "threatening litigation or legal action against a person or company. "
            "Ordinary requests for order assistance, refunds, or explanations "
            "of store policy are allowed."
        ),
    }

    topics = []
    for topic in config.GUARDRAIL_BLOCKED_TOPICS:
        definition = topic_definitions.get(topic)
        if definition is None:
            raise ValueError(
                f"No guardrail definition configured for topic: {topic!r}"
            )

        topics.append(
            {
                "name": topic,
                "definition": definition,
                "type": "DENY",
                "inputAction": "BLOCK",
                "outputAction": "BLOCK",
                "inputEnabled": True,
                "outputEnabled": True,
            }
        )

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=(
            "NovaMart customer-support guardrail for harmful content, "
            "sensitive information, restricted topics, and profanity."
        ),
        blockedInputMessaging=(
            "I can help with NovaMart orders, returns, shipping, and store "
            "policies, but I can't assist with that request."
        ),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't provide that response. I can help with "
            "NovaMart orders, returns, shipping, and store policies instead."
        ),
        contentPolicyConfig={
            "filtersConfig": filters,
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": pii_entities,
        },
        topicPolicyConfig={
            "topicsConfig": topics,
            "tierConfig": {
                "tierName": "STANDARD",
            },
        },
        wordPolicyConfig={
            "managedWordListsConfig": [
                {
                    "type": "PROFANITY",
                    "inputAction": "BLOCK",
                    "outputAction": "BLOCK",
                    "inputEnabled": True,
                    "outputEnabled": True,
                }
            ],
        },
        crossRegionConfig={
            "guardrailProfileIdentifier": "us.guardrail.v1:0",
        },
    )

    guardrail_id = response["guardrailId"]
    print(f"Created guardrail: {guardrail_id}")

    # Create a numbered version for use by the runtime.
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Version 1 of the NovaMart project guardrail",
    )
    guardrail_version = version_response["version"]

    print(
        f"Guardrail created and versioned: "
        f"{guardrail_id} (version {guardrail_version})"
    )

    return guardrail_id, guardrail_version


    # TODO: Create the guardrail
    # Use bedrock_client.create_guardrail() with:
    #   - name (config.GUARDRAIL_NAME) and description
    #   - contentPolicyConfig - filtersConfig for SEXUAL, VIOLENCE, HATE at HIGH
    #     strength and INSULTS, MISCONDUCT at MEDIUM strength (input + output)
    #   - sensitiveInformationPolicyConfig - piiEntitiesConfig:
    #       CREDIT_DEBIT_CARD_NUMBER and US_SOCIAL_SECURITY_NUMBER -> BLOCK
    #       EMAIL and PHONE -> ANONYMIZE
    #   - topicPolicyConfig - one DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS
    #     (competitor products, pricing negotiations, legal threats)
    #     Use topicPolicyConfig.tierConfig = {'tierName': 'STANDARD'} and
    #     top-level crossRegionConfig = {'guardrailProfileIdentifier': 'us.guardrail.v1:0'}.
    #     Define pricing negotiations as haggling / changing an advertised price,
    #     excluding arithmetic using an already-specified price and discount.
    #     Classic-tier definitions tested in this project blocked the math scenario.
    #     Validate allowed arithmetic (input and output) and blocked negotiation,
    #     competitor and legal-threat requests. Keep all required safety policies.
    #   - wordPolicyConfig - managedWordListsConfig with type PROFANITY
    #   - blockedInputMessaging and blockedOutputsMessaging
    #
    # Then promote it from DRAFT to a numbered version with
    # bedrock_client.create_guardrail_version(guardrailIdentifier=...)
    # and return (guardrail_id, guardrail_version).

def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str,
) -> str:
    """Deploy the orchestrator using the provided AgentCore CLI helper."""
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(
        f"  AWS Account: {config.ACCOUNT_ID}  |  "
        f"Region: {config.AWS_REGION}"
    )
    print(
        f"  Runtime: {runtime_name}  |  "
        f"CLI project: agentcore/agentcore.json "
        f"(stack {agentcore_cli.stack_name()})"
    )

    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Existing runtime will be updated: {previous_arn}")

    # Stage the application files before configuring the runtime.
    agentcore_cli.stage_runtime_code()

    runtime_env = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": guardrail_version,
    }

    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )

    agentcore_cli.deploy()

    runtime_arn = agentcore_cli.deployed_runtime_arn()
    if not runtime_arn:
        raise RuntimeError(
            "AgentCore CLI deployment finished, but no runtime ARN "
            "was recorded. Check the deployment output and CLI state."
        )

    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end="", flush=True)

    wait_for_runtime_ready(
        agentcore_control,
        runtime_arn.split("/")[-1],
    )

    print(" ready.")
    return runtime_arn

# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create or reuse AgentCore Memory with seven-day event retention
    and a session-summary strategy.

    Returns:
        The memory resource ARN.
    """
    memory_name = config.MEMORY_NAME

    # Find an existing memory, including resources on subsequent pages.
    response = agentcore_control.list_memories()

    while True:
        for existing in response.get("memories", []):
            if existing.get("id", "").startswith(memory_name):
                memory_arn = existing["arn"]
                print(f"AgentCore Memory already exists: {memory_arn}")
                return memory_arn

        next_token = response.get("nextToken")
        if not next_token:
            break

        response = agentcore_control.list_memories(
            nextToken=next_token
        )

    # Create a memory with seven-day event retention and session summaries.
    response = agentcore_control.create_memory(
        name=memory_name,
        description=(
            "NovaMart customer-support conversational memory with "
            "session summaries retained for seven days."
        ),
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "SessionSummary",
                    "namespaces": [
                        "/summaries/{actorId}/{sessionId}"
                    ],
                }
            }
        ],
        clientToken=str(uuid.uuid4()),
    )

    memory = response["memory"]
    print(
        f"Memory created: {memory['arn']} "
        f"(status: {memory['status']})"
    )
    print("Waiting for memory status ACTIVE", end="", flush=True)

    deadline = time.time() + 300

    while memory.get("status") != "ACTIVE":
        if memory.get("status") == "FAILED":
            raise RuntimeError(
                "AgentCore Memory creation failed: "
                f"{memory.get('failureReason', 'No reason provided')}"
            )

        if time.time() >= deadline:
            raise TimeoutError(
                "AgentCore Memory did not become ACTIVE within 300 seconds. "
                f"Last status: {memory.get('status')}"
            )

        time.sleep(10)
        print(".", end="", flush=True)

        memory = agentcore_control.get_memory(
            memoryId=memory["id"]
        )["memory"]

    print(" ready.")
    return memory["arn"]


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════


def configure_observability(runtime_arn: str) -> None:
    """
    Configure CloudWatch logging and X-Ray tracing for the deployed agent.
    """
    from agent_observability import apply_observability_config

    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }

    try:
        summary = apply_observability_config(
            runtime_arn,
            logging_configuration,
        )

        print(
            f"  CloudWatch logging enabled: "
            f"{summary.get('log_group', config.AGENT_LOG_GROUP)}"
        )

        xray_summary = summary.get("xray", {})
        print(
            f"  X-Ray tracing enabled: "
            f"{xray_summary.get('destination', 'configured')}"
        )
        print("  X-Ray sampling rate: 100%")

    except Exception as exc:
        print(f"[Note] Observability configuration failed: {exc}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
