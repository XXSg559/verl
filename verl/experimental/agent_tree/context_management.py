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
Context management for tree-based multi-agent systems.

This module provides intelligent context compression and information fusion
to handle the exponential growth of context in tree-structured rollouts.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextCompressionConfig:
    """Configuration for context compression strategies."""

    max_context_length: int = 1024
    compression_ratio: float = 0.5  # Target compression ratio
    preserve_keywords: List[str] = None
    agent_type_weights: Dict[str, float] = None
    depth_scaling_factor: float = 0.8  # Reduce context budget by this factor per depth

    def __post_init__(self):
        if self.preserve_keywords is None:
            self.preserve_keywords = [
                "solution", "answer", "result", "conclusion", "therefore",
                "equation", "formula", "algorithm", "step", "method"
            ]

        if self.agent_type_weights is None:
            self.agent_type_weights = {
                "mathematical": 1.2,
                "logical": 1.1,
                "evaluator": 1.3,
                "creative": 0.9,
                "general": 1.0
            }


@dataclass
class NodeContext:
    """Structured context information for a tree node."""

    # Core content
    compressed_context: str  # Compressed representation of path to root
    local_generation: str    # Only this node's contribution
    agent_insights: Dict[str, str]  # Key insights from different agent types

    # Metadata
    context_length: int
    compression_ratio: float
    preserved_keywords: List[str]
    agent_contributions: Dict[str, int]  # Track token contributions per agent type

    # Path information
    depth: int
    path_summary: str
    key_decisions: List[str]


class ContextCompressor(ABC):
    """Abstract base class for context compression strategies."""

    @abstractmethod
    async def compress_context(
        self,
        messages: List[Dict[str, Any]],
        config: ContextCompressionConfig
    ) -> str:
        """Compress a list of messages into a shorter context."""
        pass

    @abstractmethod
    def extract_key_information(
        self,
        content: str,
        agent_type: str = "general"
    ) -> List[str]:
        """Extract key information based on agent type."""
        pass


class SimpleContextCompressor(ContextCompressor):
    """
    Lightweight context compressor using extractive summarization.

    This compressor uses rule-based methods to extract key sentences
    and maintain important information while reducing context length.
    """

    def __init__(self, config: ContextCompressionConfig):
        self.config = config

        # Sentence importance weights
        self.importance_patterns = {
            "conclusion": [r"\b(therefore|thus|hence|conclusion|result)\b", 2.0],
            "solution": [r"\b(solution|answer|solve|equals|=)\b", 1.8],
            "method": [r"\b(method|approach|algorithm|step|process)\b", 1.5],
            "reasoning": [r"\b(because|since|reason|explain|why)\b", 1.3],
            "question": [r"\?\s*$", 1.2],
            "enumeration": [r"\b(first|second|next|finally|step \d+)\b", 1.4],
        }

    async def compress_context(
        self,
        messages: List[Dict[str, Any]],
        config: ContextCompressionConfig
    ) -> str:
        """Compress messages using extractive summarization."""

        if not messages:
            return ""

        # Extract all content
        all_content = []
        agent_contributions = {}

        for msg in messages:
            content = msg.get("content", "")
            agent_type = msg.get("agent_type", "general")

            all_content.append({
                "content": content,
                "agent_type": agent_type,
                "role": msg.get("role", "assistant")
            })

            agent_contributions[agent_type] = agent_contributions.get(agent_type, 0) + len(content.split())

        # Score and select important sentences
        important_sentences = self._select_important_sentences(
            all_content, config.max_context_length
        )

        # Combine into compressed context
        compressed = self._combine_sentences(important_sentences, config)

        logger.debug(f"Compressed {sum(len(ac['content'].split()) for ac in all_content)} "
                    f"tokens to {len(compressed.split())} tokens")

        return compressed

    def extract_key_information(
        self,
        content: str,
        agent_type: str = "general"
    ) -> List[str]:
        """Extract key information based on agent type and content patterns."""

        sentences = self._split_into_sentences(content)
        key_info = []

        # Agent-specific extraction patterns
        if agent_type == "mathematical":
            key_info.extend(self._extract_mathematical_info(sentences))
        elif agent_type == "logical":
            key_info.extend(self._extract_logical_info(sentences))
        elif agent_type == "creative":
            key_info.extend(self._extract_creative_info(sentences))
        elif agent_type == "evaluator":
            key_info.extend(self._extract_evaluation_info(sentences))
        else:
            key_info.extend(self._extract_general_info(sentences))

        return key_info

    def _select_important_sentences(
        self,
        content_list: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Tuple[str, float, str]]:
        """Select most important sentences within token budget."""

        sentence_scores = []

        for content_dict in content_list:
            content = content_dict["content"]
            agent_type = content_dict["agent_type"]

            sentences = self._split_into_sentences(content)

            for sentence in sentences:
                score = self._score_sentence(sentence, agent_type)
                sentence_scores.append((sentence, score, agent_type))

        # Sort by importance score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Select sentences within token budget
        selected = []
        current_tokens = 0

        for sentence, score, agent_type in sentence_scores:
            sentence_tokens = len(sentence.split())

            if current_tokens + sentence_tokens <= max_tokens:
                selected.append((sentence, score, agent_type))
                current_tokens += sentence_tokens
            else:
                # Try to fit a shortened version
                if current_tokens + sentence_tokens * 0.7 <= max_tokens:
                    shortened = self._shorten_sentence(sentence)
                    selected.append((shortened, score * 0.8, agent_type))
                    current_tokens += len(shortened.split())

        return selected

    def _score_sentence(self, sentence: str, agent_type: str) -> float:
        """Score sentence importance based on patterns and agent type."""

        base_score = 1.0

        # Apply pattern-based scoring
        for pattern_type, (pattern, weight) in self.importance_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                base_score += weight

        # Apply agent type weighting
        agent_weight = self.config.agent_type_weights.get(agent_type, 1.0)
        base_score *= agent_weight

        # Keyword preservation bonus
        for keyword in self.config.preserve_keywords:
            if keyword.lower() in sentence.lower():
                base_score += 0.5

        # Length penalty for very long sentences
        word_count = len(sentence.split())
        if word_count > 30:
            base_score *= 0.8
        elif word_count > 50:
            base_score *= 0.6

        return base_score

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences with some intelligence."""

        # Basic sentence splitting with some improvements
        sentences = re.split(r'[.!?]+\s+', content)

        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def _combine_sentences(
        self,
        selected_sentences: List[Tuple[str, float, str]],
        config: ContextCompressionConfig
    ) -> str:
        """Combine selected sentences into coherent compressed context."""

        if not selected_sentences:
            return ""

        # Group by agent type for better organization
        agent_groups = {}
        for sentence, score, agent_type in selected_sentences:
            if agent_type not in agent_groups:
                agent_groups[agent_type] = []
            agent_groups[agent_type].append(sentence)

        # Combine with agent type labels for clarity
        combined_parts = []

        for agent_type, sentences in agent_groups.items():
            if len(sentences) > 1:
                # Multiple sentences from same agent type
                agent_summary = f"[{agent_type.upper()}] " + " ".join(sentences)
            else:
                agent_summary = f"[{agent_type.upper()}] " + sentences[0]

            combined_parts.append(agent_summary)

        return " | ".join(combined_parts)

    def _shorten_sentence(self, sentence: str) -> str:
        """Shorten a sentence while preserving key information."""

        # Remove parenthetical expressions
        shortened = re.sub(r'\([^)]*\)', '', sentence)

        # Remove non-essential adverbs and adjectives
        shortened = re.sub(r'\b(very|quite|really|extremely|somewhat)\s+', '', shortened)

        # Remove redundant phrases
        shortened = re.sub(r'\b(it is clear that|it should be noted that|as we can see)\b', '', shortened)

        return shortened.strip()

    def _extract_mathematical_info(self, sentences: List[str]) -> List[str]:
        """Extract mathematical-specific information."""
        math_info = []

        for sentence in sentences:
            # Look for equations, formulas, numerical results
            if re.search(r'[=\+\-\*/]|equation|formula|\d+', sentence):
                math_info.append(sentence)
            # Look for mathematical reasoning
            elif re.search(r'solve|calculate|derive|proof|theorem', sentence, re.IGNORECASE):
                math_info.append(sentence)

        return math_info

    def _extract_logical_info(self, sentences: List[str]) -> List[str]:
        """Extract logical reasoning information."""
        logic_info = []

        for sentence in sentences:
            # Look for logical connectors and reasoning
            if re.search(r'therefore|thus|hence|because|since|if.*then|implies', sentence, re.IGNORECASE):
                logic_info.append(sentence)

        return logic_info

    def _extract_creative_info(self, sentences: List[str]) -> List[str]:
        """Extract creative and innovative information."""
        creative_info = []

        for sentence in sentences:
            # Look for creative indicators
            if re.search(r'creative|innovative|novel|unique|original|idea', sentence, re.IGNORECASE):
                creative_info.append(sentence)

        return creative_info

    def _extract_evaluation_info(self, sentences: List[str]) -> List[str]:
        """Extract evaluation and assessment information."""
        eval_info = []

        for sentence in sentences:
            # Look for evaluation indicators
            if re.search(r'quality|accuracy|correct|wrong|better|worse|score|rate', sentence, re.IGNORECASE):
                eval_info.append(sentence)

        return eval_info

    def _extract_general_info(self, sentences: List[str]) -> List[str]:
        """Extract general important information."""
        general_info = []

        for sentence in sentences:
            # Look for conclusions and summaries
            if re.search(r'conclusion|summary|result|answer|solution', sentence, re.IGNORECASE):
                general_info.append(sentence)

        return general_info


class AgentInformationFusion:
    """
    Fuses information from multiple agents with different specializations.

    This class handles the intelligent combination of insights from different
    agent types, taking into account their confidence levels and specializations.
    """

    def __init__(self, config: ContextCompressionConfig):
        self.config = config

    async def fuse_agent_insights(
        self,
        agent_outputs: Dict[str, Dict[str, Any]],
        target_specialization: str = "general"
    ) -> Dict[str, str]:
        """Fuse insights from multiple agents."""

        if not agent_outputs:
            return {}

        fused_insights = {}

        # Process each agent type
        for agent_type, output in agent_outputs.items():
            content = output.get("content", "")
            confidence = output.get("confidence", 1.0)

            # Extract key insights
            key_insights = self._extract_agent_insights(content, agent_type)

            # Weight by confidence and relevance to target
            relevance_weight = self._compute_relevance_weight(agent_type, target_specialization)
            weighted_insights = self._weight_insights(key_insights, confidence * relevance_weight)

            fused_insights[agent_type] = weighted_insights

        return fused_insights

    def _extract_agent_insights(self, content: str, agent_type: str) -> List[str]:
        """Extract key insights from agent content."""

        compressor = SimpleContextCompressor(self.config)
        return compressor.extract_key_information(content, agent_type)

    def _compute_relevance_weight(self, source_agent: str, target_agent: str) -> float:
        """Compute how relevant source agent's insights are to target agent."""

        if source_agent == target_agent:
            return 1.0

        # Define agent compatibility matrix
        compatibility = {
            ("mathematical", "logical"): 0.8,
            ("logical", "mathematical"): 0.8,
            ("evaluator", "mathematical"): 0.9,
            ("evaluator", "logical"): 0.9,
            ("evaluator", "creative"): 0.7,
            ("creative", "general"): 0.6,
            ("mathematical", "general"): 0.7,
            ("logical", "general"): 0.7,
        }

        return compatibility.get((source_agent, target_agent), 0.5)

    def _weight_insights(self, insights: List[str], weight: float) -> str:
        """Weight and combine insights based on importance."""

        if not insights:
            return ""

        # Simple weighting: select top insights and add confidence indicator
        top_insights = insights[:max(1, int(len(insights) * weight))]

        confidence_indicator = ""
        if weight > 0.8:
            confidence_indicator = "[HIGH CONFIDENCE] "
        elif weight > 0.5:
            confidence_indicator = "[MEDIUM CONFIDENCE] "
        else:
            confidence_indicator = "[LOW CONFIDENCE] "

        return confidence_indicator + " ".join(top_insights)


class DynamicContextSelector:
    """
    Dynamically selects relevant context based on current task and history.

    This class uses simple relevance scoring to select the most pertinent
    information from the tree history for the current generation step.
    """

    def __init__(self, config: ContextCompressionConfig):
        self.config = config

    async def select_relevant_context(
        self,
        current_task: str,
        historical_nodes: List[Dict[str, Any]],
        max_context_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """Select most relevant historical context for current task."""

        if max_context_tokens is None:
            max_context_tokens = self.config.max_context_length

        if not historical_nodes:
            return []

        # Score relevance of each historical node
        scored_nodes = []

        for node_data in historical_nodes:
            content = node_data.get("content", "")
            agent_type = node_data.get("agent_type", "general")

            relevance_score = self._compute_text_relevance(current_task, content, agent_type)
            scored_nodes.append((node_data, relevance_score))

        # Sort by relevance
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Select nodes within token budget
        selected_context = []
        token_count = 0

        for node_data, score in scored_nodes:
            content = node_data.get("content", "")
            content_tokens = len(content.split())

            if token_count + content_tokens <= max_context_tokens:
                selected_context.append({
                    **node_data,
                    "relevance_score": score
                })
                token_count += content_tokens
            else:
                # Try to include a summary
                if token_count + content_tokens * 0.3 <= max_context_tokens:
                    summary = self._create_node_summary(content, int(content_tokens * 0.3))
                    selected_context.append({
                        **node_data,
                        "content": summary,
                        "relevance_score": score * 0.8,
                        "is_summary": True
                    })
                    token_count += len(summary.split())

        return selected_context

    def _compute_text_relevance(self, query: str, content: str, agent_type: str) -> float:
        """Compute relevance score between query and content."""

        # Simple keyword-based relevance (can be improved with embeddings)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        # Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)

        if union == 0:
            jaccard_sim = 0.0
        else:
            jaccard_sim = intersection / union

        # Agent type relevance bonus
        agent_weight = self.config.agent_type_weights.get(agent_type, 1.0)

        # Keyword importance boost
        important_matches = 0
        for keyword in self.config.preserve_keywords:
            if keyword.lower() in query.lower() and keyword.lower() in content.lower():
                important_matches += 1

        keyword_bonus = important_matches * 0.2

        return jaccard_sim * agent_weight + keyword_bonus

    def _create_node_summary(self, content: str, target_length: int) -> str:
        """Create a brief summary of node content."""

        sentences = content.split('.')
        if len(sentences) <= 1:
            return content[:target_length * 5]  # Rough character estimation

        # Take first and last sentences as summary
        summary_parts = [sentences[0].strip()]
        if len(sentences) > 1:
            summary_parts.append(sentences[-1].strip())

        summary = ". ".join(summary_parts)

        # Truncate if still too long
        words = summary.split()
        if len(words) > target_length:
            summary = " ".join(words[:target_length]) + "..."

        return summary


class HierarchicalContextManager:
    """
    Manages context hierarchically across tree depths.

    This class coordinates the overall context management strategy,
    integrating compression, fusion, and selection components.
    """

    def __init__(self, config: ContextCompressionConfig):
        self.config = config
        self.compressor = SimpleContextCompressor(config)
        self.fusion = AgentInformationFusion(config)
        self.selector = DynamicContextSelector(config)

    async def build_node_context(
        self,
        current_node_info: Dict[str, Any],
        parent_context: Optional[str],
        sibling_nodes: List[Dict[str, Any]],
        historical_nodes: List[Dict[str, Any]]
    ) -> NodeContext:
        """Build comprehensive context for a tree node."""

        depth = current_node_info.get("depth", 0)
        agent_type = current_node_info.get("agent_type", "general")
        current_task = current_node_info.get("content", "")

        # Calculate context budget for this depth
        depth_budget = int(self.config.max_context_length * (self.config.depth_scaling_factor ** depth))

        # 1. Compress parent context
        compressed_parent = ""
        if parent_context:
            parent_messages = [{"content": parent_context, "agent_type": "previous", "role": "assistant"}]
            compressed_parent = await self.compressor.compress_context(parent_messages, self.config)

        # 2. Fuse sibling information
        sibling_insights = {}
        if sibling_nodes:
            sibling_outputs = {}
            for sibling in sibling_nodes:
                sib_type = sibling.get("agent_type", "general")
                sibling_outputs[sib_type] = {
                    "content": sibling.get("content", ""),
                    "confidence": sibling.get("confidence", 1.0),
                    "specialization": sib_type
                }

            sibling_insights = await self.fusion.fuse_agent_insights(sibling_outputs, agent_type)

        # 3. Select relevant historical context
        relevant_history = await self.selector.select_relevant_context(
            current_task, historical_nodes, max_context_tokens=depth_budget // 2
        )

        # 4. Build final context
        context_parts = []

        if compressed_parent:
            context_parts.append(f"[CONTEXT] {compressed_parent}")

        if sibling_insights:
            insights_text = " | ".join(f"{k}: {v}" for k, v in sibling_insights.items())
            context_parts.append(f"[INSIGHTS] {insights_text}")

        if relevant_history:
            history_text = " | ".join(f"[{h.get('agent_type', 'unk')}] {h['content'][:100]}..."
                                    for h in relevant_history[:3])
            context_parts.append(f"[HISTORY] {history_text}")

        final_context = " | ".join(context_parts)

        # Truncate if necessary
        context_words = final_context.split()
        if len(context_words) > depth_budget:
            final_context = " ".join(context_words[:depth_budget]) + "..."

        # Build NodeContext object
        node_context = NodeContext(
            compressed_context=final_context,
            local_generation=current_task,
            agent_insights=sibling_insights,
            context_length=len(final_context.split()),
            compression_ratio=len(final_context.split()) / max(1, len(parent_context.split()) if parent_context else 1),
            preserved_keywords=[kw for kw in self.config.preserve_keywords if kw.lower() in final_context.lower()],
            agent_contributions={agent_type: len(current_task.split())},
            depth=depth,
            path_summary=compressed_parent,
            key_decisions=self._extract_key_decisions(final_context)
        )

        return node_context

    def _extract_key_decisions(self, context: str) -> List[str]:
        """Extract key decision points from context."""

        decision_patterns = [
            r'decide[ds]?\s+to\s+([^.]+)',
            r'choose[s]?\s+([^.]+)',
            r'select[s]?\s+([^.]+)',
            r'therefore\s+([^.]+)',
            r'conclusion[:\s]*([^.]+)'
        ]

        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            decisions.extend(matches[:2])  # Limit to avoid clutter

        return decisions[:5]  # Maximum 5 key decisions