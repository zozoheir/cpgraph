import asyncio
import logging
from typing import Dict

from cpgraph.graph import Node, CPGraph
from sortino_backend.data.cache.redis_client import redis_client

logging.basicConfig(level=logging.INFO)


class SampleNode(Node):
    async def run(self, first_input) -> Dict:
        first_input = first_input + 1
        print(first_input)
        await asyncio.sleep(0.1)
        return {f"first_input": first_input}


async def main():
    # Create graph

    graph = CPGraph(name="StateDemo",
                    redis_client=redis_client)

    # Create nodes
    nodes = [SampleNode() for _ in range(5)]

    # Setup dependencies
    graph.add_node(nodes[0])
    graph.add_node(nodes[1], depends_on=[nodes[0]])
    graph.add_node(nodes[2], depends_on=[nodes[0]])
    graph.add_node(nodes[3], depends_on=[nodes[1], nodes[2]])
    graph.add_node(nodes[4], depends_on=[nodes[3]])

    # Visualize initial state
    print("Initial state:")

    # Execute graph
    await graph.run(first_input=1)

    graph.draw(layout='spring')


if __name__ == "__main__":
    asyncio.run(main())