#
# Sydney Bites AI Host - Gemini Live Restaurant Voice Agent
# Fixed for current Pipecat + SmallWebRTC
#

import os
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)


async def check_table_availability(params):
    date = params.get("date", "today")
    time = params.get("time", "7pm")
    party_size = int(params.get("party_size", 2))
    logger.info(f"Table check: {party_size} people at {time} on {date}")
    return {
        "available": True,
        "message": f"Yes, we have tables available for {party_size} at {time} on {date}.",
    }


async def save_order(params):
    items = params.get("items", [])
    total = float(params.get("total", 0.0))
    name = params.get("customer_name", "Guest")
    logger.info(f"🍔 ORDER: {items} | Total: ${total:.2f} | Name: {name}")
    return {
        "status": "confirmed",
        "message": "Order confirmed! We'll get that ready shortly.",
    }


system_instruction = """
You are Sydney Bites AI Host — a friendly, casual receptionist at a busy cafe in Sydney, Australia.

Greet with: "G'day! Sydney Bites here, how can I help you today?"

- Take orders naturally
- Answer menu questions, suggest items, handle dietary requests
- Check table availability when asked
- Confirm the full order before saving it
- Be relaxed and interruptible
- Handle noisy backgrounds well

Menu:
- Classic Beef Burger — $18
- Halloumi Burger (v) — $17
- Chicken Schnitzel — $22
- Margherita Pizza — $19
- Caesar Salad — $16
- Fries — $8
Drinks, beer & wine available.
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("🚀 Starting Sydney Bites AI Host...")

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GeminiLiveLLMService.Settings(
            system_instruction=system_instruction,
        ),
    )

    llm.register_function("check_table_availability", check_table_availability)
    llm.register_function("save_order", save_order)

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        user_agg,
        llm,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Greet the customer warmly and offer help with orders or bookings."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    # Define transport params for webrtc (this was missing)
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()