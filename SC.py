import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Parameters for the network
num_nodes = np.random.randint(2, 21)
num_ports = np.random.randint(1, 3)
num_plants = num_nodes - num_ports

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
for i in range(num_nodes):
    G.add_node(i, type='plant' if i < num_plants else 'port')

# Connect each plant to at least one port
for plant in range(num_plants):
    target_port = np.random.choice(range(num_plants, num_nodes))
    G.add_edge(plant, target_port)

# Add additional edges to ports
for port in range(num_plants, num_nodes):
    # Each port connects to other ports or plants
    for _ in range(np.random.randint(1, 4)):
        potential_targets = list(range(num_plants)) + [p for p in range(num_plants, num_nodes) if p != port]
        target = np.random.choice(potential_targets)
        G.add_edge(port, target)

# Visualization
colors = ['blue' if G.nodes[node]['type'] == 'port' else 'green' for node in G.nodes]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=colors)
plt.show()
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Parameters for the network
num_nodes = np.random.randint(2, 21)
num_ports = np.random.randint(1, 3)
num_plants = num_nodes - num_ports

# Generate a power-law degree sequence
exponent = np.random.uniform(2, 3)  # Power-law exponent
degree_sequence = nx.utils.powerlaw_sequence(num_nodes, exponent)

# Ensure the degrees are integers and adjust for feasibility
degree_sequence = [max(1, int(np.round(d))) for d in degree_sequence]

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
for i in range(num_nodes):
    G.add_node(i, type='plant' if i < num_plants else 'port')

# Function to add edges based on degree sequence
def add_edges_following_degree_sequence(G, degree_sequence, num_plants):
    degrees = dict(zip(G.nodes(), degree_sequence))
    while sum(degrees.values()) > 0:
        plant_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'plant' and degrees[n] > 0]
        port_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'port' and degrees[n] > 0]

        # Select random plant and port
        plant = np.random.choice(plant_nodes)
        port = np.random.choice(port_nodes)

        # Add edge if not exists
        if not G.has_edge(plant, port):
            G.add_edge(plant, port)
            degrees[plant] -= 1
            degrees[port] -= 1

# Add edges
add_edges_following_degree_sequence(G, degree_sequence, num_plants)

# Visualization
colors = ['blue' if G.nodes[node]['type'] == 'port' else 'green' for node in G.nodes]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color=colors)
plt.show()