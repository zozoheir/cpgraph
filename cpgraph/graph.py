"""
Production-ready async compute graph with state management and visualization
Features:
- Node state machine (INIT, RUNNING, SUCCESS, FAILED)
- State transition validation
- NetworkX-based visualization
- Execution tracking with color coding
"""
import asyncio
import json
import logging
import pprint
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum, auto
from typing import Dict, List, Optional
from uuid import uuid4

import redis
from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel

from cpgraph.redis_encoder import RedisEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s -  %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class State(Enum):
    """Node state machine states"""
    INIT = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()


class Node(ABC):
    """Base Node class with state management and async execution"""

    save_cache = True
    fetch_cache = True
    ttl = 3600
    model = None

    def __init__(self, node_id: Optional[str] = None):
        self.id = node_id or self.__class__.__name__ + f"_{uuid4()}"
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.id[-4:]}]")
        self.state = State.INIT

    def transition_state(self, new_state: State):
        """Validate and update node state"""
        self.logger.debug(f"State change: {self.state.name} → {new_state.name}")
        self.state = new_state

    @abstractmethod
    async def run(self, **kwargs) -> Dict:
        """Execute node's logic and return output as a dictionary"""
        return await self._run(**kwargs)

    @observe()
    async def _run(self, **kwargs) -> Dict:
        """Execute node's logic and return output as a dictionary"""
        langfuse_context.update_current_observation(
            name=self.__class__.__name__,
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.id} ({self.state.name})>"


class InvalidStateTransitionError(Exception):
    """Invalid node state transition attempt"""


class CPGraph:
    """Async compute graph with visualization and state tracking"""

    def __init__(self,
                 name,
                 redis_client: redis.Redis,
                 stop_on_exception=True):
        self.redis_client = redis_client
        self.id = str(uuid4())
        self.name = name
        self.nodes = {}
        self.dependencies = defaultdict(list)
        self.reverse_dependencies = defaultdict(set)
        self.execution_data = None
        self._topology_valid = False
        self.stop_on_exception = stop_on_exception

    def add_node(self, node: Node, depends_on: Optional[List[Node]] = None):
        """Add node with ordered dependencies"""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists in graph")

        self.nodes[node.id] = node
        self.dependencies[node.id] = []

        if depends_on:
            for dependency in depends_on:
                if dependency.id not in self.nodes:
                    raise ValueError(f"Dependency {dependency} not in graph")
                self.dependencies[node.id].append(dependency.id)
                self.reverse_dependencies[dependency.id].add(node.id)

        self._topology_valid = False

    def _validate_topology(self):
        """Check for cycles and single final node"""
        visited = set()
        recursion_stack = set()

        def visit(node_id):
            if node_id in recursion_stack:
                raise ValueError(f"Cycle detected at node {node_id}")
            if node_id in visited:
                return

            visited.add(node_id)
            recursion_stack.add(node_id)

            for neighbor in self.reverse_dependencies[node_id]:
                visit(neighbor)

            recursion_stack.remove(node_id)

        for node_id in self.nodes:
            visit(node_id)

        # Validate single final node
        final_nodes = [nid for nid in self.nodes
                       if not self.reverse_dependencies[nid]]
        if len(final_nodes) != 1:
            final_nodes_ = [self.nodes[i] for i in final_nodes]

            raise ValueError(
                f"Graph must have exactly one final node, found {len(final_nodes)}: {final_nodes_}"
            )

        self._topology_valid = True

    def get_execution_order(self) -> List[List[Node]]:
        """Get topological execution order with parallel batches"""
        if not self._topology_valid:
            self._validate_topology()

        in_degree = defaultdict(int)
        queue = deque()
        execution_order = []

        for node_id in self.nodes:
            in_degree[node_id] = len(self.dependencies[node_id])
            if in_degree[node_id] == 0:
                queue.append(node_id)

        while queue:
            current_batch = []
            for _ in range(len(queue)):
                node_id = queue.popleft()
                current_batch.append(self.nodes[node_id])

                for neighbor in self.reverse_dependencies[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            execution_order.append(current_batch)

        return execution_order

    @observe(as_type='span')
    async def run(self, **initial_kwargs) -> Dict:
        langfuse_context.update_current_observation(
            name=self.name
        )

        # Validation and state transitions
        self._validate_topology()
        execution_order = self.get_execution_order()
        for node in self.nodes.values():
            node.transition_state(State.INIT)
        self.execution_data = {
            nid: {'input': None, 'output': None}
            for nid in self.nodes
        }

        # Prepare batches
        for batch in execution_order:
            batch_inputs = {}
            for node in batch:
                deps = self.dependencies[node.id]
                input_kwargs = initial_kwargs.copy()  # Always start with initial kwargs

                # Merge outputs from all dependencies
                for dep_id in deps:
                    if self.execution_data[dep_id]['output']:
                        input_kwargs.update(self.execution_data[dep_id]['output'])

                batch_inputs[node.id] = input_kwargs


            # Execute batch
            results = await asyncio.gather(*[
                self._execute_node(node, batch_inputs[node.id])
                for node in batch
            ], return_exceptions=True)

            # Store results in execution data
            for node, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.execution_data[node.id]['output'] = None
                    if self.stop_on_exception:
                        raise result
                else:
                    self.execution_data[node.id]['output'] = result

        return self._get_final_output()

    @observe(as_type='span')
    async def _execute_node(self, node: Node, input_context: Dict) -> Dict:
        """Execute single node with state tracking"""

        langfuse_context.update_current_observation(
            name=node.__class__.__name__,
        )
        if self and any(node.state == State.FAILED for node in self.nodes.values()):
            return None

        try:
            node.transition_state(State.RUNNING)
            node.logger.info(f"Starting execution")

            # caching
            kwargs_items = sorted(input_context.items())
            cache_key_kwargs = ','.join(f"{k}={v}" for k, v in kwargs_items)
            cache_key = f"({cache_key_kwargs})".replace(':', '_').replace(' ', '_')
            cache_key = f"cpgraph:.{node.__class__.__name__}:{cache_key}"

            if node.fetch_cache:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    if node.model:
                        node_output_dict = {**node.model(**json.loads(cached_data)).__dict__}
                        node.logger.info(f"Completed successfully. ")
                        node.transition_state(State.SUCCESS)
                        output = node_output_dict
                    else:
                        output = json.loads(cached_data)
                    node.logger.info(f"Completed successfully. Returned cache. ")
                    node.transition_state(State.SUCCESS)
                    return output

            # Run node
            output = await node.run(**input_context.copy())
            node.logger.info(f"Completed successfully. ")
            node.transition_state(State.SUCCESS)

            if node.save_cache and output is not None:
                if node.ttl:
                    self.redis_client.setex(cache_key, node.ttl, json.dumps(output, cls=RedisEncoder))
                else:
                    self.redis_client.set(cache_key, json.dumps(output, cls=RedisEncoder))

            return output
        except Exception as e:
            node.transition_state(State.FAILED)
            node.logger.error(f"Failed during execution: {pprint.pformat(traceback.format_exception(e))}")

    def draw(self):
        """
        Visualize the graph using NetworkX and matplotlib
        Colors nodes by state: gray=INIT, blue=RUNNING, green=SUCCESS, red=FAILED
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.warning("Visualization requires networkx and matplotlib")
            return

        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()

        # Build graph structure
        for node_id in self.nodes:
            G.add_node(node_id)
        for node_id, deps in self.dependencies.items():
            for dep_id in deps:
                G.add_edge(dep_id, node_id)

        # Create visualization elements
        color_map = {
            State.INIT: '#CCCCCC',
            State.RUNNING: '#1f78b4',
            State.SUCCESS: '#33a02c',
            State.FAILED: '#e31a1c'
        }
        colors = [color_map[node.state] for node in self.nodes.values()]
        labels = {nid: f"{node.__class__.__name__}"
                  for nid, node in self.nodes.items()}

        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        nx.draw(G, pos, labels=labels, node_color=colors, with_labels=True, edge_color="gray", node_size=2000,
                font_size=10)

        plt.title(f"Execution Graph: {self.id}")
        plt.show()

    def _get_final_output(self) -> Dict:
        """Retrieve output from final node"""
        final_node_id = next(nid for nid in self.nodes
                             if not self.reverse_dependencies[nid])
        return self.execution_data[final_node_id]['output'].model_dump() if isinstance(
            self.execution_data[final_node_id]['output'], BaseModel) else self.execution_data[final_node_id]['output']
