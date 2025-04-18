import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define region and pin nodes
regions = [str(i) for i in range(1, 9)]
pin_region = {
    "x1": "1", "x2": "2", "x3": "4", "x4": "4", "x5": "2",
    "x6": "8", "x7": "5", "x8": "5", "x9": "6", "x10": "7"
}
pins = list(pin_region.keys())

# Create graph
G = nx.Graph()

# Add region nodes and edges
G.add_nodes_from(regions)
region_edges = [
    ("1", "2"), ("1", "4"),
    ("2", "4"), ("2", "5"), ("2", "8"),
    ("3", "4"), ("3", "6"), ("3", "7"),
    ("4", "5"),
    ("5", "6"), ("5", "8"),
    ("6", "7"),
    ("7", "8")
]
G.add_edges_from(region_edges)

# Add pin nodes and edges
for pin, region in pin_region.items():
    G.add_node(pin)
    G.add_edge(pin, region)

# Circular layout for region nodes
circle_pos = nx.circular_layout(G.subgraph(regions), scale=5.0)

# Group pins by region
pins_by_region = defaultdict(list)
for pin, region in pin_region.items():
    pins_by_region[region].append(pin)

# Spread out pins around each region
pin_pos = {}
offset = 1.8  # Radial distance of pins
angle_offset = np.pi / 12  # Angular separation between pins
for region, pins in pins_by_region.items():
    x, y = circle_pos[region]
    base_angle = np.arctan2(y, x)
    num_pins = len(pins)
    for i, pin in enumerate(pins):
        # Center the cluster of pins around the base angle
        angle = base_angle + angle_offset * (i - (num_pins - 1) / 2)
        pin_pos[pin] = (
            x + offset * np.cos(angle),
            y + offset * np.sin(angle)
        )

# Combine region and pin positions
pos = {**circle_pos, **pin_pos}

# Verify pin list from the actual graph
pins = [n for n in G.nodes if n.startswith("x")]

# Draw
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, nodelist=regions, node_color="skyblue", node_size=1200)
nx.draw_networkx_nodes(G, pos, nodelist=pins, node_color="violet", node_size=800)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, font_size=12)

plt.title("Region Adjacency Graph with Spaced Pins", fontsize=14)
plt.axis("off")
plt.show()

