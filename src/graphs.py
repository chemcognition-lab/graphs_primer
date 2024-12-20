import dataclasses
import json
from pathlib import Path

import dataclasses_json
import networkx as nx
import networkx.readwrite


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class FeatureSpec:
    name: str
    data_type: str  # continuos, binary, categorical
    dim: int
    values: list[str] | tuple[float, float]

    def is_discrete(self):
        return self.data_type == "categorical"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphSpec:
    nodes: list[FeatureSpec]
    edges: list[FeatureSpec]
    globals: list[FeatureSpec]
    nodes_are: str
    edges_are: str
    name: str
    short_name: str


def save_as_json(g: nx.Graph, spec: GraphSpec, filename: str) -> None:
    with Path(filename).open("w") as f:
        data = nx.readwrite.json_graph.node_link_data(g)
        data["spec"] = spec.to_dict()
        json.dump(data, f)


def load_from_json(g, filename) -> tuple[nx.Graph, GraphSpec]:
    with Path(filename).open("r") as f:
        data = json.load(f)
        spec = GraphSpec.from_dict(data.pop("spec"))
        g = nx.readwrite.json_graph.node_link_graph(data)
    return g, spec
