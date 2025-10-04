# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AsyncBatchVLLMModel provides async batch generation capabilities for tree-structured rollouts.

This module extends BatchVLLMModel with full async support, enabling both agent-level
and step-level batch generation through async/await patterns.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .batch_vllm_model import BatchVLLMModel

logger = logging.getLogger(__name__)

try:
    from smolagents.models import ChatMessage, MessageRole
    from smolagents.monitoring import TokenUsage
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    ChatMessage = None
    MessageRole = None
    TokenUsage = None

try:
    import vllm
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
except (ImportError, ValueError) as e:
    VLLM_AVAILABLE = False
    SamplingParams = None
    vllm = None


class AsyncBatchVLLMModel:
    """
    Async-enabled batch VLLMModel for high-performance parallel generation.

    This class provides both synchronous and asynchronous interfaces for batch generation,
    enabling efficient step-level and agent-level batch processing in tree structures.

    Key features:
    - Full async/await support for non-blocking generation
    - Intelligent batch size management to avoid OOM
    - Concurrent generation with configurable parallelism
    - Backward compatibility with sync interfaces
    - Advanced error handling and retry mechanisms
    """

    def __init__(
        self,
        model_id: str,
        batch_size: int = 3,
        max_concurrent_batches: int = 2,
        max_batch_size: int = 16,
        model_kwargs: Optional[Dict[str, Any]] = None,
        executor_max_workers: Optional[int] = None,
        **kwargs
    ):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is required but not available")
        if not VLLM_AVAILABLE:
            raise ImportError("vllm is required but not available")

        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.max_batch_size = max_batch_size
        self.model_kwargs = model_kwargs or {}

        # Initialize base BatchVLLMModel for sync compatibility
        self.base_model = BatchVLLMModel(
            model_id=model_id,
            batch_size=batch_size,
            model_kwargs=self.model_kwargs,
            **kwargs
        )

        # Async execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=executor_max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)

        # Performance tracking
        self.total_async_generations = 0
        self.total_async_batch_size = 0
        self.async_generation_times = []

        logger.info(f"AsyncBatchVLLMModel initialized: batch_size={batch_size}, max_concurrent={max_concurrent_batches}")

    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if hasattr(self.base_model, 'cleanup'):
            self.base_model.cleanup()

    # Sync interface compatibility
    def generate(self, messages: List[ChatMessage | dict], **kwargs) -> ChatMessage:
        """Synchronous single generation for compatibility."""
        return self.base_model.generate(messages, **kwargs)

    def generate_batch(self, messages: List[ChatMessage | dict], n: Optional[int] = None, **kwargs) -> List[ChatMessage]:
        """Synchronous batch generation for compatibility."""
        return self.base_model.generate_batch(messages, n=n, **kwargs)

    # Async interface
    async def generate_async(
        self,
        messages: List[ChatMessage | dict],
        **kwargs
    ) -> ChatMessage:
        """
        Asynchronous single generation.

        Args:
            messages: Input messages for generation
            **kwargs: Additional generation parameters

        Returns:
            Single ChatMessage with generated content
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.base_model.generate,
            messages,
            **kwargs
        )

    async def generate_batch_async(
        self,
        batch_messages: List[List[ChatMessage | dict]],
        n: Optional[int] = None,
        **kwargs
    ) -> List[ChatMessage]:
        """
        Asynchronous batch generation for multiple prompt sets.

        This is the core method for step-level batch processing where each
        element in batch_messages represents a different agent's prompt.

        Args:
            batch_messages: List of message lists, one per agent/branch
            n: Number of generations per prompt (defaults to 1 for step-level)
            **kwargs: Additional generation parameters

        Returns:
            List of ChatMessage objects, one per input prompt
        """
        if not batch_messages:
            return []

        n = n or 1
        effective_batch_size = len(batch_messages)

        logger.debug(f"Async batch generation: {effective_batch_size} prompts, n={n}")

        # Use semaphore to limit concurrent batches
        async with self.semaphore:
            start_time = time.time()

            try:
                if effective_batch_size <= self.max_batch_size:
                    # Single batch processing
                    results = await self._process_single_batch_async(batch_messages, n, **kwargs)
                else:
                    # Split into multiple batches to avoid OOM
                    results = await self._process_large_batch_async(batch_messages, n, **kwargs)

                # Update metrics
                generation_time = time.time() - start_time
                self.total_async_generations += 1
                self.total_async_batch_size += effective_batch_size
                self.async_generation_times.append(generation_time)

                logger.debug(f"Async batch completed: {effective_batch_size} results in {generation_time:.2f}s")
                return results

            except Exception as e:
                logger.error(f"Async batch generation failed: {e}")
                raise

    async def _process_single_batch_async(
        self,
        batch_messages: List[List[ChatMessage | dict]],
        n: int,
        **kwargs
    ) -> List[ChatMessage]:
        """Process a single batch that fits within max_batch_size."""
        # For step-level batching, we need to generate one response per agent
        # So we create individual tasks for each agent's prompt
        tasks = []

        for messages in batch_messages:
            task = self._generate_single_async(messages, **kwargs)
            tasks.append(task)

        # Execute all generations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Generation {i} failed: {result}")
                # Create error placeholder
                error_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"Generation failed: {str(result)}",
                    raw={"error": str(result), "batch_index": i}
                )
                processed_results.append(error_message)
            else:
                processed_results.append(result)

        return processed_results

    async def _process_large_batch_async(
        self,
        batch_messages: List[List[ChatMessage | dict]],
        n: int,
        **kwargs
    ) -> List[ChatMessage]:
        """Process a large batch by splitting into smaller chunks."""
        all_results = []

        for i in range(0, len(batch_messages), self.max_batch_size):
            chunk = batch_messages[i:i + self.max_batch_size]
            chunk_results = await self._process_single_batch_async(chunk, n, **kwargs)
            all_results.extend(chunk_results)

        return all_results

    async def _generate_single_async(
        self,
        messages: List[ChatMessage | dict],
        **kwargs
    ) -> ChatMessage:
        """Generate a single response asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.base_model.generate,
            messages,
            **kwargs
        )

    # Advanced async methods for specialized use cases
    async def generate_step_synchronized_async(
        self,
        agent_prompts: Dict[str, List[ChatMessage | dict]],
        **kwargs
    ) -> Dict[str, ChatMessage]:
        """
        Generate responses for multiple agents synchronously at the same step.

        This method is specifically designed for step-level batch processing
        where multiple agents need to proceed together.

        Args:
            agent_prompts: Dict mapping agent_id to their current prompt
            **kwargs: Generation parameters

        Returns:
            Dict mapping agent_id to their generated response
        """
        agent_ids = list(agent_prompts.keys())
        prompts = list(agent_prompts.values())

        # Batch generate all responses
        responses = await self.generate_batch_async(prompts, n=1, **kwargs)

        # Return as dict mapping agent_id to response
        return dict(zip(agent_ids, responses))

    async def generate_agent_branches_async(
        self,
        base_prompt: List[ChatMessage | dict],
        n: int,
        **kwargs
    ) -> List[ChatMessage]:
        """
        Generate multiple branches for agent-level batch processing.

        Args:
            base_prompt: The base prompt to branch from
            n: Number of branches to generate
            **kwargs: Generation parameters

        Returns:
            List of n different responses for agent-level branching
        """
        # For agent-level batching, we use the base model's batch generation
        # which generates n different responses to the same prompt
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.base_model.generate_batch,
            base_prompt,
            n,
            **kwargs
        )

    def get_async_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for async operations."""
        if not self.async_generation_times:
            return {"async_operations": 0}

        avg_time = sum(self.async_generation_times) / len(self.async_generation_times)
        avg_batch_size = self.total_async_batch_size / self.total_async_generations if self.total_async_generations > 0 else 0

        return {
            "async_operations": self.total_async_generations,
            "total_async_batch_size": self.total_async_batch_size,
            "average_async_time": avg_time,
            "average_async_batch_size": avg_batch_size,
            "max_concurrent_batches": self.max_concurrent_batches,
            "max_batch_size": self.max_batch_size
        }

    def __getattr__(self, name):
        """Delegate unknown attributes to base model for compatibility."""
        return getattr(self.base_model, name)