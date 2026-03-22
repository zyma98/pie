"""
Knowledge graph extraction and querying using inferlib (Python).

Has the LLM extract entity-relation triples from a text passage, builds a
directed graph using a simple adjacency list, then uses the graph structure
to provide context for answering a follow-up question.

Demonstrates overlapping GPU and CPU work: the query context's system prompt
is submitted for prefill while the graph is being built on the CPU side.
"""

from collections import deque

from inference_bindings import (
    Context,
    Model,
    SamplerConfig_Greedy,
    StopConfig,
    set_return,
)
from run_bindings import get_arguments

PASSAGE = (
    "France is a country in Western Europe. Paris is the capital of France. "
    "The Eiffel Tower is a landmark located in Paris. France borders Germany to the east. "
    "Berlin is the capital of Germany. The Brandenburg Gate is a landmark in Berlin. "
    "Germany borders Poland to the east. Warsaw is the capital of Poland. "
    "The Palace of Culture and Science is a landmark in Warsaw. "
    "France is a member of the European Union. Germany is a member of the European Union. "
    "Poland is a member of the European Union. The European Union is headquartered in Brussels. "
    "Brussels is the capital of Belgium. Belgium borders France to the south."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge extraction assistant. Given a text passage, extract factual "
    "relationships as triples.\n\n"
    'Output format: start with the line "RELATIONS:" followed by one triple per line '
    "in the exact format:\n"
    "subject | relation | object\n\n"
    "Rules:\n"
    '- Use consistent entity names (e.g. always "France", not "france" or "the country of France")\n'
    "- Each triple should capture a single factual relationship\n"
    "- Do not output anything after the last triple"
)

QUERY_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using provided knowledge graph data. "
    "You will receive a list of facts extracted from a knowledge graph. Use only these facts "
    "to answer the question. Be concise."
)

QUESTION = "What landmarks can you find in the capitals of EU member countries?"


class DirectedGraph:
    """Simple directed graph with string nodes and edge labels."""

    def __init__(self) -> None:
        self._nodes: dict[str, int] = {}
        self._node_labels: list[str] = []
        self._outgoing: list[list[tuple[int, str]]] = []
        self._incoming: list[list[tuple[int, str]]] = []

    def _get_or_add(self, name: str) -> int:
        if name in self._nodes:
            return self._nodes[name]
        idx = len(self._node_labels)
        self._nodes[name] = idx
        self._node_labels.append(name)
        self._outgoing.append([])
        self._incoming.append([])
        return idx

    def add_edge(self, src: str, relation: str, dst: str) -> None:
        s = self._get_or_add(src)
        d = self._get_or_add(dst)
        self._outgoing[s].append((d, relation))
        self._incoming[d].append((s, relation))

    def node_count(self) -> int:
        return len(self._node_labels)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._outgoing)

    def entity_names(self) -> list[str]:
        return list(self._nodes.keys())

    def retrieve_facts(self, seed_entities: list[str], depth: int) -> list[str]:
        """BFS from seed entities, collecting facts up to given depth."""
        visited: set[int] = set()
        queue: deque[tuple[int, int]] = deque()
        facts: list[str] = []

        for entity in seed_entities:
            if entity in self._nodes:
                idx = self._nodes[entity]
                if idx not in visited:
                    visited.add(idx)
                    queue.append((idx, 0))

        while queue:
            node_idx, current_depth = queue.popleft()
            entity = self._node_labels[node_idx]

            for target_idx, relation in self._outgoing[node_idx]:
                target = self._node_labels[target_idx]
                facts.append(f"{entity} {relation} {target}")
                if current_depth + 1 < depth and target_idx not in visited:
                    visited.add(target_idx)
                    queue.append((target_idx, current_depth + 1))

            for source_idx, relation in self._incoming[node_idx]:
                source = self._node_labels[source_idx]
                facts.append(f"{source} {relation} {entity}")
                if current_depth + 1 < depth and source_idx not in visited:
                    visited.add(source_idx)
                    queue.append((source_idx, current_depth + 1))

        return facts


def extract_relations_section(text: str) -> str:
    idx = text.rfind("RELATIONS:")
    if idx >= 0:
        return text[idx + len("RELATIONS:"):].strip()
    return text.strip()


def parse_triples(text: str) -> list[tuple[str, str, str]]:
    section = extract_relations_section(text)
    triples: list[tuple[str, str, str]] = []
    for line in section.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def poll_flush(ctx: Context) -> None:
    """Block until a flush_async completes."""
    future = ctx.flush_async()
    if future is not None:
        pollable = future.pollable()
        pollable.block()
        del pollable
        del future


def main() -> None:
    args = get_arguments()
    max_tokens = int(args.get("max_tokens", "2048"))
    depth = int(args.get("depth", "3"))

    model = Model.get_auto()
    eos_tokens = model.eos_tokens()
    stop_config = StopConfig(max_tokens=max_tokens, eos_sequences=eos_tokens)

    # --- Stage 1: Extract triples from the passage ---
    print("--- Stage 1: Extracting knowledge triples ---")

    extraction_ctx = Context(model)
    extraction_ctx.fill_system(EXTRACTION_SYSTEM_PROMPT)
    extraction_ctx.fill_user(
        f"Extract all factual triples from this passage:\n\n{PASSAGE}"
    )

    extraction_output = extraction_ctx.generate(SamplerConfig_Greedy(), stop_config)
    print(f"Extraction output: {extraction_output}")

    # Start prefilling the query context's system prompt on the GPU
    # while we do CPU-bound graph construction below.
    query_ctx = Context(model)
    query_ctx.fill_system(QUERY_SYSTEM_PROMPT)
    poll_flush(query_ctx)

    # --- Stage 2: Parse triples and build the knowledge graph ---
    print("\n--- Stage 2: Building knowledge graph ---")

    triples = parse_triples(extraction_output)
    print(f"Extracted {len(triples)} triples:")
    for subj, rel, obj in triples:
        print(f"  {subj} | {rel} | {obj}")

    graph = DirectedGraph()
    for subj, rel, obj in triples:
        graph.add_edge(subj, rel, obj)

    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"Entities: {', '.join(graph.entity_names())}")

    # --- Stage 3: Query the graph for relevant context ---
    print(f'\n--- Stage 3: Querying graph (depth={depth}) for: "{QUESTION}" ---')

    query_entities = ["European Union"]
    all_facts = graph.retrieve_facts(query_entities, depth)
    all_facts = sorted(set(all_facts))

    print(f"Retrieved {len(all_facts)} relevant facts:")
    for fact in all_facts:
        print(f"  - {fact}")

    # --- Stage 4: Answer the question using graph context ---
    print("\n--- Stage 4: Generating answer ---")

    facts_text = "\n".join(f"- {f}" for f in all_facts)
    query_ctx.fill_user(
        f"Knowledge graph facts:\n{facts_text}\n\nQuestion: {QUESTION}"
    )

    answer = query_ctx.generate(SamplerConfig_Greedy(), stop_config)
    print(f"Answer: {answer}")

    set_return(answer)


if __name__ == "__main__":
    main()
