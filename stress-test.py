#!/usr/bin/env python
# coding: utf-8

# In[29]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Function to add a plant with warehousing cost
def add_plant_with_warehousing_cost(G, node_name, capacity):
    theta = random.uniform(1, 3)  # Scale parameter for the gamma distribution
    warehousing_cost = np.random.gamma(1, theta)  # k=1, theta varies between 1 and 3
    total_cost = capacity * warehousing_cost  # Total cost based on capacity * unit warehousing cost
    G.add_node(node_name, level=1, type='Plant', capacity=capacity, warehousing_cost=warehousing_cost, total_cost=total_cost)

    
# Function to add transit time to edges
def add_transit_time(G):
    for u, v in G.edges():
        b = random.uniform(2, 3)  # Varying b between 2 and 3 for the triangular distribution
        transit_time = np.random.triangular(left=1, mode=b, right=3)
        G[u][v]['transit_time'] = transit_time
        
# Function to add transportation cost to edges
def add_transportation_cost(G):
    for u, v in G.edges():
        mu = random.uniform(1, np.log(25))  # Vary mu between 1 and ln(25)
        sigma = random.uniform(np.log(5), np.log(25))  # Vary sigma between ln(5) and ln(25)
        transportation_cost = np.random.lognormal(mu, sigma)
        G[u][v]['transportation_cost'] = transportation_cost
        
# Initialize the corrected graph
G_corrected = nx.DiGraph()

# Define the initial number of nodes
initial_ports = 1
initial_plants = 2
initial_customers = 3

# Add initial nodes with warehousing cost for plants
for i in range(initial_ports):
    G_corrected.add_node(f"Port_{i+1}", level=0, type='Port')
for i in range(initial_plants):
    capacity = np.abs(int(np.random.normal(random.uniform(0, 50), random.uniform(25, 50))))
    add_plant_with_warehousing_cost(G_corrected, f"Plant_{i+1}", capacity)
for i in range(initial_customers):
    G_corrected.add_node(f"Customer_{i+1}", level=2, type='Customer', demand=np.abs(int(np.random.normal(random.uniform(0, 50), random.uniform(25, 50)))))

# Create initial connectivity
G_corrected.add_edge("Port_1", "Plant_1")
G_corrected.add_edge("Plant_1", "Customer_1")
G_corrected.add_edge("Port_1", "Plant_2")
G_corrected.add_edge("Plant_2", "Customer_2")
G_corrected.add_edge("Plant_2", "Customer_3")

# Function to add nodes with preferential attachment based on node type
def add_node_with_corrected_rules(G, node_type):
    new_node_id = len(G.nodes) + 1
    new_node_name = f"{node_type}_{new_node_id}"
    
    if node_type == "Plant":
        capacity = np.abs(int(np.random.normal(random.uniform(0, 50), random.uniform(25, 50))))
        add_plant_with_warehousing_cost(G, new_node_name, capacity)
        # Connect to a port
        port_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'Port']
        chosen_port = random.choice(port_nodes)
        G.add_edge(chosen_port, new_node_name)
    elif node_type == "Customer":
        demand = np.abs(int(np.random.normal(random.uniform(0, 5), random.uniform(5, 10))))
        G.add_node(new_node_name, level=2, type='Customer', demand=demand)
        # Connect to a plant
        plant_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'Plant']
        chosen_plant = random.choice(plant_nodes)
        G.add_edge(chosen_plant, new_node_name)
    elif node_type == "Port":
        G.add_node(new_node_name, level=0, type='Port')

# Add additional nodes
num_additional_nodes = 500  # Total additional nodes to add
for _ in range(num_additional_nodes):
    node_type_choice = np.random.choice(["Port", "Plant", "Customer"], p=[0.1, 0.4, 0.5])
    add_node_with_corrected_rules(G_corrected, node_type_choice)

# Calculate accumulated demand for plants and ports

for node in G_corrected.nodes(data=True):
    if node[1]['type'] == 'Plant':
        # Sum the demand of directly connected customers to this plant
        connected_customers = [n for n in G_corrected.successors(node[0])]
        G_corrected.nodes[node[0]]['accumulated_demand'] = sum(G_corrected.nodes[customer]['demand'] for customer in connected_customers if 'demand' in G_corrected.nodes[customer])

# Correctly calculate accumulated demand for ports
for node in G_corrected.nodes(data=True):
    if node[1]['type'] == 'Port':
        # Sum the accumulated demand of all plants connected to this port
        connected_plants = [n for n in G_corrected.successors(node[0])]
        G_corrected.nodes[node[0]]['accumulated_demand'] = sum(G_corrected.nodes[plant]['accumulated_demand'] for plant in connected_plants if 'accumulated_demand' in G_corrected.nodes[plant])


# Add transit time to edges after creating the graph
add_transit_time(G_corrected)        
        

# Add transportation cost to edges after creating the graph
add_transportation_cost(G_corrected)

# Visualization with all corrections applied
plt.figure(figsize=(14, 12))
pos_corrected = nx.spring_layout(G_corrected)

nx.draw_networkx_nodes(G_corrected, pos_corrected, node_size=700, node_color=[{'Port': 'red', 'Plant': 'green', 'Customer': 'blue'}[G_corrected.nodes[node]['type']] for node in G_corrected.nodes])

# Display transit times and transportation costs on edges
edge_labels = {(u, v): f"Transit Time: {G_corrected[u][v]['transit_time']:.1f}\nTransportation Cost: {G_corrected[u][v]['transportation_cost']:.2f}" for u, v in G_corrected.edges()}
nx.draw_networkx_edge_labels(G_corrected, pos_corrected, edge_labels=edge_labels, font_size=8)

nx.draw_networkx_edges(G_corrected, pos_corrected, arrows=True)


# Updated labels to include warehousing cost and total cost for plants, and correct accumulated demands
labels_with_all_info = {}
for node, data in G_corrected.nodes(data=True):
    label_lines = [f"{node}"]  # Start with the node name
    if data['type'] == 'Customer':
        # For customers, include only the demand
        label_lines.append(f"Demand: {data['demand']}")
    elif data['type'] == 'Plant':
        # For plants, include capacity, warehousing cost, total cost, and accumulated demand
        label_lines.append(f"Capacity: {data['capacity']}")
        label_lines.append(f"Warehousing Cost: {data['warehousing_cost']:.2f}")
        label_lines.append(f"Total Cost: {data['total_cost']:.2f}")
        label_lines.append(f"Acc. Demand: {data['accumulated_demand']}")
    elif data['type'] == 'Port':
        # For ports, include accumulated demand
        label_lines.append(f"Acc. Demand: {data['accumulated_demand']}")

    labels_with_all_info[node] = "\n".join(label_lines)  # Join all label lines


nx.draw_networkx_labels(G_corrected, pos_corrected, labels=labels_with_all_info, font_size=8)

plt.axis('off')
plt.show()


from math import log

# Function to calculate the normalized frequency of each unique value
def calculate_frequencies(values):
    total = sum(values)
    frequencies = [v / total for v in values]
    return frequencies

# Calculate H(C) - Entropy of Capacity across all plants
capacities = [G_corrected.nodes[node]['capacity'] for node in G_corrected.nodes if G_corrected.nodes[node]['type'] == 'Plant']
total_capacity = sum(capacities)
frequency_capacities = calculate_frequencies(capacities)
entropy_capacity = -sum(f * log(f, 2) for f in frequency_capacities if f > 0)

print(f"Entropy of H(C) across all plants: {entropy_capacity:.4f}")

# Calculate H(D_i) for each plant and compute the updated vulnerability index v_p(i)
vulnerability_index = {}

for node in G_corrected.nodes:
    if G_corrected.nodes[node]['type'] == 'Plant':
        # Extracting demands of connected customers to this plant
        connected_customers = [n for n in G_corrected.successors(node)]
        demands = [G_corrected.nodes[customer]['demand'] for customer in connected_customers]
        total_demand = sum(demands)
        frequency_demands = calculate_frequencies(demands)
        entropy_demand = -sum(f * log(f, 2) for f in frequency_demands if f > 0)

        # Compute the updated vulnerability index for the plant using the absolute value
        vp_i = abs((entropy_capacity - entropy_demand) / entropy_capacity) if entropy_capacity != 0 else 0
        vulnerability_index[node] = vp_i

# Print the updated vulnerability index of each plant node
print("Updated Vulnerability Index of Each Plant Node:")
for plant, v in vulnerability_index.items():
    print(f"{plant}: {v:.4f}")
    
    
    
from math import log
import numpy as np

def calculate_entropy_weighted(values):
    """Calculate entropy assuming the values come from a normal distribution."""
    # Ensure there are multiple values to calculate variance
    if len(values) > 1:
        variance = np.var(values, ddof=1)  # Sample variance
    else:
        return 0  # Entropy calculation not applicable for single value

    # Calculate entropy of a normal distribution
    entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
    return entropy


# Calculate H(C x W_p)
product_capacity_warehousing = [G_corrected.nodes[node]['capacity'] * G_corrected.nodes[node]['warehousing_cost'] for node in G_corrected.nodes if G_corrected.nodes[node]['type'] == 'Plant']
entropy_capacity_warehousing = calculate_entropy_weighted(product_capacity_warehousing)

warehouse_cost_vulnerability = {}

for node in G_corrected.nodes:
    if G_corrected.nodes[node]['type'] == 'Plant':
        W_pi = G_corrected.nodes[node]['warehousing_cost']
        C_i = G_corrected.nodes[node]['capacity']
        connected_customers = [n for n in G_corrected.successors(node)]
        
        # Calculate individual adjustments for each customer
        adjusted_capacities = [(C_i - G_corrected.nodes[customer]['demand']) * W_pi for customer in connected_customers]
        
        # Calculate entropy for these adjusted capacities
        entropy_adjusted_capacity_warehousing = calculate_entropy_weighted(adjusted_capacities)
        
        # Assuming entropy_capacity_warehousing is calculated globally before this loop

        v_W_p_i = abs((entropy_capacity_warehousing - entropy_adjusted_capacity_warehousing) / entropy_capacity_warehousing)
        
        
        warehouse_cost_vulnerability[node] = v_W_p_i

# Print the warehouse cost vulnerability of each plant node
print("Warehouse Cost Vulnerability of Each Plant Node:")
for plant, v in warehouse_cost_vulnerability.items():
    print(f"{plant}: {v:.4f}")    

    
from math import log

def calculate_entropy(values):
    """Calculate entropy from a list of values, safely handling log(0)."""
    total = sum(values)
    if total == 0:
        return 0  # If the total is 0, entropy is undefined or considered to be 0.
    frequencies = [v / total for v in values]
    entropy = -sum(f * log(f) if f > 0 else 0 for f in frequencies)  # Only compute log(f) if f > 0.
    return entropy

# Calculate global entropy using all edges
all_demands_times_costs = [
    G_corrected.edges[edge]['transit_time'] * G_corrected.nodes[edge[0]]['accumulated_demand']
    for edge in G_corrected.edges
    if 'accumulated_demand' in G_corrected.nodes[edge[0]] and 'transit_time' in G_corrected.edges[edge]  # Use accumulated_demand from plants
]
global_entropy = calculate_entropy(all_demands_times_costs)

# Initialize a dictionary to hold the transportation time vulnerability for each port node
vulnerability_C_T = {}

for node in G_corrected.nodes:
    if G_corrected.nodes[node]['type'] == 'Port':
        port_links_products = []


        for u, v, data in G_corrected.edges(data=True):
            if u == node:  # Ensuring edge ends at the port
                if 'transit_time' in data and 'accumulated_demand' in G_corrected.nodes[u]:  # Check conditions
                    product = data['transit_time'] * G_corrected.nodes[u]['accumulated_demand']
                    port_links_products.append(product)

        if not port_links_products:
            continue

        local_entropy = calculate_entropy(port_links_products)

        # Calculate vulnerability; ensure global_entropy is not zero to avoid division by zero
        if global_entropy > 0:
            vulnerability = (global_entropy - local_entropy) / global_entropy 
        else:
            vulnerability = 0

        vulnerability_C_T[node] = vulnerability
        
        
# Print the transportation time vulnerability of each port node
print("Transportation Time Vulnerability of Each Port Node:")
for port_node, vulnerability in vulnerability_C_T.items():
    print(f"{port_node}: {vulnerability:.4f}")
    
    
    
def calculate_transportation_cost_vulnerability(G_corrected):
    # Step 1: Adjust transportation time for each edge
    Te = {(u, v): min(data['transit_time'], 2) for u, v, data in G_corrected.edges(data=True)}

    # Step 2: Calculate global entropy
    global_products = [G_corrected.nodes[u]['accumulated_demand'] * Te[(u, v)] for u, v in G_corrected.edges() if 'accumulated_demand' in G_corrected.nodes[u]]
    global_entropy = calculate_entropy(global_products)

    vulnerability_C_T = {}  # Transportation cost vulnerability for each node

    # Step 3: Calculate local entropy for each node
    for node in G_corrected.nodes:
        if G_corrected.nodes[node]['type'] == 'Port':  # Assuming we're calculating this for ports, adjust as needed
            local_products = [G_corrected.nodes[u]['accumulated_demand'] * Te[(u, v)] for u, v in G_corrected.edges() if u == node and 'accumulated_demand' in G_corrected.nodes[u]]
            local_entropy = calculate_entropy(local_products)

            # Step 4: Compute v_T(i') for the node

            vulnerability = (global_entropy - local_entropy) / global_entropy


            vulnerability_C_T[node] = vulnerability

    return vulnerability_C_T

# Calculate transportation cost vulnerability
vulnerability_C_T = calculate_transportation_cost_vulnerability(G_corrected)

# Print the transportation cost vulnerability of each port node
print("Transportation Cost Vulnerability of Each Port Node:")
for port_node, vulnerability in vulnerability_C_T.items():
    print(f"{port_node}: {vulnerability:.4f}")

    
    


# In[33]:





def run_simulation():
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import random

# Function to add a plant with warehousing cost
    def add_plant_with_warehousing_cost(G, node_name, capacity):
        theta = random.uniform(1, 3)  # Scale parameter for the gamma distribution
        warehousing_cost = np.random.gamma(1, theta)  # k=1, theta varies between 1 and 3
        total_cost = capacity * warehousing_cost  # Total cost based on capacity * unit warehousing cost
        G.add_node(node_name, level=1, type='Plant', capacity=capacity, warehousing_cost=warehousing_cost, total_cost=total_cost)

    
    # Function to add transit time to edges
    def add_transit_time(G):
        for u, v in G.edges():
            b = random.uniform(2, 3)  # Varying b between 2 and 3 for the triangular distribution
            transit_time = np.random.triangular(left=1, mode=b, right=3)
            G[u][v]['transit_time'] = transit_time

    # Function to add transportation cost to edges
    def add_transportation_cost(G):
        for u, v in G.edges():
            mu = random.uniform(1, np.log(25))  # Vary mu between 1 and ln(25)
            sigma = random.uniform(np.log(5), np.log(25))  # Vary sigma between ln(5) and ln(25)
            transportation_cost = np.random.lognormal(mu, sigma)
            G[u][v]['transportation_cost'] = transportation_cost

    # Initialize the corrected graph
    G_corrected = nx.DiGraph()

    # Define the initial number of nodes
    initial_ports = 1
    initial_plants = 2
    initial_customers = 3

    # Add initial nodes with warehousing cost for plants
    for i in range(initial_ports):
        G_corrected.add_node(f"Port_{i+1}", level=0, type='Port')
    for i in range(initial_plants):
        capacity = np.abs(int(np.random.normal(random.uniform(25, 50), random.uniform(25, 50))))
        add_plant_with_warehousing_cost(G_corrected, f"Plant_{i+1}", capacity)
    for i in range(initial_customers):
        G_corrected.add_node(f"Customer_{i+1}", level=2, type='Customer', demand=np.abs(int(np.random.normal(random.uniform(0, 50), random.uniform(25, 50)))))

    # Create initial connectivity
    G_corrected.add_edge("Port_1", "Plant_1")
    G_corrected.add_edge("Plant_1", "Customer_1")
    G_corrected.add_edge("Port_1", "Plant_2")
    G_corrected.add_edge("Plant_2", "Customer_2")
    G_corrected.add_edge("Plant_2", "Customer_3")

    # Function to add nodes with preferential attachment based on node type
    def add_node_with_corrected_rules(G, node_type):
        new_node_id = len(G.nodes) + 1
        new_node_name = f"{node_type}_{new_node_id}"

        if node_type == "Plant":
            capacity = np.abs(int(np.random.normal(random.uniform(25, 50), random.uniform(25, 50))))
            add_plant_with_warehousing_cost(G, new_node_name, capacity)
            # Connect to a port
            port_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'Port']
            chosen_port = random.choice(port_nodes)
            G.add_edge(chosen_port, new_node_name)
        elif node_type == "Customer":
            demand = np.abs(int(np.random.normal(random.uniform(0, 5), random.uniform(5, 10))))
            G.add_node(new_node_name, level=2, type='Customer', demand=demand)
            # Connect to a plant
            plant_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'Plant']
            chosen_plant = random.choice(plant_nodes)
            G.add_edge(chosen_plant, new_node_name)
        elif node_type == "Port":
            G.add_node(new_node_name, level=0, type='Port')

    # Add additional nodes
    num_additional_nodes = 15  # Total additional nodes to add
    for _ in range(num_additional_nodes):
        node_type_choice = np.random.choice(["Port", "Plant", "Customer"], p=[0.1, 0.4, 0.5])
        add_node_with_corrected_rules(G_corrected, node_type_choice)

    # Calculate accumulated demand for plants and ports

    for node in G_corrected.nodes(data=True):
        if node[1]['type'] == 'Plant':
            # Sum the demand of directly connected customers to this plant
            connected_customers = [n for n in G_corrected.successors(node[0])]
            G_corrected.nodes[node[0]]['accumulated_demand'] = sum(G_corrected.nodes[customer]['demand'] for customer in connected_customers if 'demand' in G_corrected.nodes[customer])

    # Correctly calculate accumulated demand for ports
    for node in G_corrected.nodes(data=True):
        if node[1]['type'] == 'Port':
            # Sum the accumulated demand of all plants connected to this port
            connected_plants = [n for n in G_corrected.successors(node[0])]
            G_corrected.nodes[node[0]]['accumulated_demand'] = sum(G_corrected.nodes[plant]['accumulated_demand'] for plant in connected_plants if 'accumulated_demand' in G_corrected.nodes[plant])


    # Add transit time to edges after creating the graph
    add_transit_time(G_corrected)        


    # Add transportation cost to edges after creating the graph
    add_transportation_cost(G_corrected)

    # Visualization with all corrections applied
    pos_corrected = nx.spring_layout(G_corrected)

    # Display transit times and transportation costs on edges
    edge_labels = {(u, v): f"Transit Time: {G_corrected[u][v]['transit_time']:.1f}\nTransportation Cost: {G_corrected[u][v]['transportation_cost']:.2f}" for u, v in G_corrected.edges()}
    #nx.draw_networkx_edge_labels(G_corrected, pos_corrected, edge_labels=edge_labels, font_size=8)

    # Updated labels to include warehousing cost and total cost for plants, and correct accumulated demands
    labels_with_all_info = {}
    for node, data in G_corrected.nodes(data=True):
        label_lines = [f"{node}"]  # Start with the node name
        if data['type'] == 'Customer':
            # For customers, include only the demand
            label_lines.append(f"Demand: {data['demand']}")
        elif data['type'] == 'Plant':
            # For plants, include capacity, warehousing cost, total cost, and accumulated demand
            label_lines.append(f"Capacity: {data['capacity']}")
            label_lines.append(f"Warehousing Cost: {data['warehousing_cost']:.2f}")
            label_lines.append(f"Total Cost: {data['total_cost']:.2f}")
            label_lines.append(f"Acc. Demand: {data['accumulated_demand']}")
        elif data['type'] == 'Port':
            # For ports, include accumulated demand
            label_lines.append(f"Acc. Demand: {data['accumulated_demand']}")

        labels_with_all_info[node] = "\n".join(label_lines)  # Join all label lines


    #nx.draw_networkx_labels(G_corrected, pos_corrected, labels=labels_with_all_info, font_size=8)




    from math import log

    # Function to calculate the normalized frequency of each unique value
    def calculate_frequencies(values):
        total = sum(values)
        frequencies = [v / total for v in values]
        return frequencies

    # Calculate H(C) - Entropy of Capacity across all plants
    capacities = [G_corrected.nodes[node]['capacity'] for node in G_corrected.nodes if G_corrected.nodes[node]['type'] == 'Plant']
    total_capacity = sum(capacities)
    frequency_capacities = calculate_frequencies(capacities)
    entropy_capacity = -sum(f * log(f, 2) for f in frequency_capacities if f > 0)


    # Calculate H(D_i) for each plant and compute the updated vulnerability index v_p(i)
    vulnerability_index = {}

    for node in G_corrected.nodes:
        if G_corrected.nodes[node]['type'] == 'Plant':
            # Extracting demands of connected customers to this plant
            connected_customers = [n for n in G_corrected.successors(node)]
            demands = [G_corrected.nodes[customer]['demand'] for customer in connected_customers]
            total_demand = sum(demands)
            frequency_demands = calculate_frequencies(demands)
            entropy_demand = -sum(f * log(f, 2) for f in frequency_demands if f > 0)

            # Compute the updated vulnerability index for the plant using the absolute value
            vp_i = abs((entropy_capacity - entropy_demand) / entropy_capacity) if entropy_capacity != 0 else 0
            vulnerability_index[node] = vp_i




    from math import log
    import numpy as np

    def calculate_entropy_weighted(values):
        """Calculate entropy assuming the values come from a normal distribution."""
        # Ensure there are multiple values to calculate variance
        if len(values) > 1:
            variance = np.var(values, ddof=1)  # Sample variance
        else:
            return 0  # Entropy calculation not applicable for single value

        # Calculate entropy of a normal distribution
        entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
        return entropy


    # Calculate H(C x W_p)
    product_capacity_warehousing = [G_corrected.nodes[node]['capacity'] * G_corrected.nodes[node]['warehousing_cost'] for node in G_corrected.nodes if G_corrected.nodes[node]['type'] == 'Plant']
    entropy_capacity_warehousing = calculate_entropy_weighted(product_capacity_warehousing)

    warehouse_cost_vulnerability = {}

    for node in G_corrected.nodes:
        if G_corrected.nodes[node]['type'] == 'Plant':
            W_pi = G_corrected.nodes[node]['warehousing_cost']
            C_i = G_corrected.nodes[node]['capacity']
            connected_customers = [n for n in G_corrected.successors(node)]

            # Calculate individual adjustments for each customer
            adjusted_capacities = [(C_i - G_corrected.nodes[customer]['demand']) * W_pi for customer in connected_customers]

            # Calculate entropy for these adjusted capacities
            entropy_adjusted_capacity_warehousing = calculate_entropy_weighted(adjusted_capacities)

            # Assuming entropy_capacity_warehousing is calculated globally before this loop

            v_W_p_i = abs((entropy_capacity_warehousing - entropy_adjusted_capacity_warehousing) / entropy_capacity_warehousing)


            warehouse_cost_vulnerability[node] = v_W_p_i




    from math import log

    def calculate_entropy(values):
        """Calculate entropy from a list of values, safely handling log(0)."""
        total = sum(values)
        if total == 0:
            return 0  # If the total is 0, entropy is undefined or considered to be 0.
        frequencies = [v / total for v in values]
        entropy = -sum(f * log(f) if f > 0 else 0 for f in frequencies)  # Only compute log(f) if f > 0.
        return entropy

    # Calculate global entropy using all edges
    all_demands_times_costs = [
        G_corrected.edges[edge]['transit_time'] * G_corrected.nodes[edge[0]]['accumulated_demand']
        for edge in G_corrected.edges
        if 'accumulated_demand' in G_corrected.nodes[edge[0]] and 'transit_time' in G_corrected.edges[edge]  # Use accumulated_demand from plants
    ]
    global_entropy = calculate_entropy(all_demands_times_costs)

    # Initialize a dictionary to hold the transportation time vulnerability for each port node
    vulnerability_C_T = {}

    for node in G_corrected.nodes:
        if G_corrected.nodes[node]['type'] == 'Port':
            port_links_products = []


            for u, v, data in G_corrected.edges(data=True):
                if u == node:  # Ensuring edge ends at the port
                    if 'transit_time' in data and 'accumulated_demand' in G_corrected.nodes[u]:  # Check conditions
                        product = data['transit_time'] * G_corrected.nodes[u]['accumulated_demand']
                        port_links_products.append(product)

            if not port_links_products:
                continue

            local_entropy = calculate_entropy(port_links_products)

            # Calculate vulnerability; ensure global_entropy is not zero to avoid division by zero
            if global_entropy > 0:
                vulnerability = (global_entropy - local_entropy) / global_entropy 
            else:
                vulnerability = 0

            vulnerability_C_T[node] = vulnerability





    def calculate_transportation_cost_vulnerability(G_corrected):
        # Step 1: Adjust transportation time for each edge
        Te = {(u, v): min(data['transit_time'], 2) for u, v, data in G_corrected.edges(data=True)}

        # Step 2: Calculate global entropy
        global_products = [G_corrected.nodes[u]['accumulated_demand'] * Te[(u, v)] for u, v in G_corrected.edges() if 'accumulated_demand' in G_corrected.nodes[u]]
        global_entropy = calculate_entropy(global_products)

        vulnerability_C_T = {}  # Transportation cost vulnerability for each node

        # Step 3: Calculate local entropy for each node
        for node in G_corrected.nodes:
            if G_corrected.nodes[node]['type'] == 'Port':  # Assuming we're calculating this for ports, adjust as needed
                local_products = [G_corrected.nodes[u]['accumulated_demand'] * Te[(u, v)] for u, v in G_corrected.edges() if u == node and 'accumulated_demand' in G_corrected.nodes[u]]
                local_entropy = calculate_entropy(local_products)

                # Step 4: Compute v_T(i') for the node

                vulnerability = (global_entropy - local_entropy) / global_entropy


                vulnerability_C_T[node] = vulnerability

        return vulnerability_C_T

    # Calculate transportation cost vulnerability
    vulnerability_C_T = calculate_transportation_cost_vulnerability(G_corrected)


    # Initialize containers for the aggregated data
    demands = {}
    capacities = {}
    accumulated_demands = {}
    warehousing_costs = {}
    transportation_costs = {}
    transportation_times = {}

    # Collect data from nodes
    for node, data in G_corrected.nodes(data=True):
        if data['type'] == 'Plant':
            capacities[node] = data.get('capacity', 0)
            warehousing_costs[node] = data.get('warehousing_cost', 0)
            accumulated_demands[node] = data.get('accumulated_demand', 0)
        elif data['type'] == 'Customer':
            demands[node] = data.get('demand', 0)

    for u, v, data in G_corrected.edges(data=True):
        transportation_costs[(u, v)] = data.get('transportation_cost', 0)
        transportation_times[(u, v)] = data.get('transit_time', 0)
        

    return {
        'vulnerability_index': vulnerability_index,  # Assuming previously calculated
        'warehouse_cost_vulnerability': warehouse_cost_vulnerability,  # Assuming previously calculated
        'transportation_time_vulnerability': vulnerability_C_T,  # Assuming previously calculated
        'transportation_cost_vulnerability': calculate_transportation_cost_vulnerability(G_corrected),  # Assuming this function is defined elsewhere
        'demands': demands,
        'capacities': capacities,
        'accumulated_demands': accumulated_demands,
        'warehousing_costs': warehousing_costs,
        'transportation_costs': transportation_costs,
        'transportation_times': transportation_times
    }


# In[58]:


results = []  # List to store results of each simulation

num_simulations = 1000
for i in range(num_simulations):
    print(f"Running simulation {i+1}/{num_simulations}")
    simulation_result = run_simulation()
    results.append(simulation_result)


# In[68]:


import matplotlib.pyplot as plt
from collections import Counter

# Assuming 'results' is a list of dictionaries, each containing the simulation outcome
# and 'accumulated_demands' and 'vulnerability_index' are properly calculated for each plant

# Prepare data for plotting
capacity = []
accumulated_demands = []
warehouse_cost_vulnerability = []
pairs = []

# Iterate through the results to collect data
for result in results:
    # Access accumulated_demands and capacity separately
    for node, node_data in result['accumulated_demands'].items():
        # Ensure node exists in other dictionaries before accessing
        if node in result['warehouse_cost_vulnerability'] and node in result['capacities']:
            accumulated_demand = node_data
            warehouse_cost_vulnerability = result['warehouse_cost_vulnerability'][node]
            capacity = result['capacities'][node]

            # Only append data with vulnerability_index <= 1
            if warehouse_cost_vulnerability <= 1:
                if  accumulated_demand <= 200:
                    pairs.append((accumulated_demand, warehouse_cost_vulnerability, capacity))



# Calculate frequencies of each (accumulated_demand, vulnerability_index) pair
pair_frequencies = Counter(pairs)



# Unzip the pairs and their frequencies to separate lists
unique_pairs, frequencies = zip(*pair_frequencies.items())
unique_accumulated_demands, unique_warehouse_cost_vulnerability, unique_capacities = zip(*unique_pairs)
frequencies = [frequency * 10 for frequency in frequencies]  # Adjust size for visibility

# Normalize capacity values for coloring
norm = plt.Normalize(min(unique_capacities), max(unique_capacities))
colors = plt.cm.viridis(norm(unique_capacities))  # Map normalized capacity to a colormap

 # Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(unique_accumulated_demands, unique_warehouse_cost_vulnerability, unique_capacities, c=colors, s=frequencies, alpha=0.5)

# Create a colorbar to show the capacity scale
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
cbar.set_label('Capacity')

ax.set_title('V_wp vs. Demand vs. Capacity of Plants')
ax.set_xlabel('Demand')
ax.set_ylabel('V_wp')
ax.set_zlabel('Capacity')
plt.show()



# In[66]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import Counter

# Assuming 'results' is a list of dictionaries, each containing the simulation outcome
# and 'accumulated_demands', 'vulnerability_index', and 'capacity' are properly calculated for each plant

# Prepare data for plotting
accumulated_demands = []
vulnerability_indexes = []
capacities = []  # Rename for clarity
pairs = []

# Iterate through the results to collect data
for result in results:
    for node, node_data in result['accumulated_demands'].items():
        if node in result['vulnerability_index'] and node in result['capacities']:
            accumulated_demand = node_data
            vulnerability_index = result['vulnerability_index'][node]
            capacity_value = result['capacities'][node]  # Use a different variable name to avoid confusion

            # Only append data with vulnerability_index <= 1
            if vulnerability_index <= 1:
                if accumulated_demand <= 200:
                    pairs.append((accumulated_demand, vulnerability_index, capacity_value))
                    capacities.append(capacity_value)  # Collect capacities for coloring

# Calculate frequencies of each (accumulated_demand, vulnerability_index, capacity) pair
pair_frequencies = Counter(pairs)

# Unzip the pairs and their frequencies to separate lists
unique_pairs, frequencies = zip(*pair_frequencies.items())
unique_accumulated_demands, unique_vulnerability_indexes, unique_capacities = zip(*unique_pairs)
frequencies = [frequency * 10 for frequency in frequencies]  # Adjust size for visibility

# Normalize capacity values for coloring
norm = plt.Normalize(min(unique_capacities), max(unique_capacities))
colors = plt.cm.viridis(norm(unique_capacities))  # Map normalized capacity to a colormap

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(unique_accumulated_demands, unique_vulnerability_indexes, unique_capacities, c=colors, s=frequencies, alpha=0.5)

# Create a colorbar to show the capacity scale
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
cbar.set_label('Capacity')

ax.set_title('V_p vs. Demand vs. Capacity of Plants')
ax.set_xlabel('Demand')
ax.set_ylabel('V_p')
ax.set_zlabel('Capacity')
plt.show()


# In[89]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming `results` is a list of dictionaries from your simulations, each containing the 'transportation_costs' and 'transportation_cost_vulnerability' data
transportation_costs = []  # Store average transportation costs for ports
vulnerabilities = []  # Store transportation cost vulnerabilities for ports

for result in results:
    # For each simulation, calculate the average transportation cost associated with each port and collect the port's vulnerability
    for port, vulnerability in result['transportation_cost_vulnerability'].items():
        # Assuming transportation_costs are stored in a dictionary with keys as edge tuples (u, v)
        port_costs = [cost for (u, v), cost in result['transportation_costs'].items() if u == port or v == port]
        if port_costs:  # Check if there are any costs associated with this port
            avg_cost = np.mean(port_costs)
            if avg_cost < 20000:  # Only consider costs lower than 10000
                transportation_costs.append(avg_cost)
                vulnerabilities.append(vulnerability)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(transportation_costs, vulnerabilities, alpha=0.6)
plt.title('Transportation Cost Vulnerability vs. Average Transportation Cost')
plt.xlabel('Average Transportation Cost')
plt.ylabel('Transportation Cost Vulnerability')
plt.grid(True)
plt.show()


# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for plotting
vulnerability_indexes = []
warehousing_cost_vulnerabilities = []

# Iterate through the results to collect data
for result in results:
    # Check if the keys exist in the result dictionary
    if 'vulnerability_index' in result and 'warehouse_cost_vulnerability' in result:
        for node in result['vulnerability_index']:
            # Ensure the node is present in both dictionaries
            if node in result['warehouse_cost_vulnerability']:
                vulnerability_index = result['vulnerability_index'][node]
                warehousing_cost_vulnerability = result['warehouse_cost_vulnerability'][node]
                
                # Append the data for plotting
                if vulnerability_index <= 1:
                    if warehousing_cost_vulnerability <= 1:
                        vulnerability_indexes.append(vulnerability_index)
                        warehousing_cost_vulnerabilities.append(warehousing_cost_vulnerability)

# Plotting
plt.figure(figsize=(10, 6))
# Scatter plot
plt.scatter(vulnerability_indexes, warehousing_cost_vulnerabilities, alpha=0.6, label='Data Points')
# KDE plot
sns.kdeplot(x=vulnerability_indexes, y=warehousing_cost_vulnerabilities, levels=20, color="k", linewidths=1.5, alpha=0.5, label='KDE')
plt.title('v_p vs. v_Wp')
plt.xlabel('v_p')
plt.ylabel('v_wp')
plt.grid(True)
plt.legend()
plt.show()


# In[99]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for KDE plot

# Assuming `results` is a list of dictionaries from your simulations, each containing the 'transportation_costs' and 'transportation_cost_vulnerability' data
transportation_times = []  # Store average transportation costs for ports
vulnerabilities = []  # Store transportation cost vulnerabilities for ports

for result in results:
    # For each simulation, calculate the average transportation cost associated with each port and collect the port's vulnerability
    for port, vulnerability in result['transportation_time_vulnerability'].items():
        # Assuming transportation_costs are stored in a dictionary with keys as edge tuples (u, v)
        port_times = [time for (u, v), time in result['transportation_times'].items() if u == port or v == port]
        if port_times:  # Check if there are any costs associated with this port
            avg_time = np.mean(port_times)
            if avg_time < 20000:  # Only consider costs lower than 20000
                transportation_times.append(avg_time)
                vulnerabilities.append(vulnerability)

# Plotting
plt.figure(figsize=(10, 6))
# Scatter plot
plt.scatter(transportation_times, vulnerabilities, alpha=0.6, label='Data Points')
# KDE plot
sns.kdeplot(x=transportation_times, y=vulnerabilities, levels=20, color="k", linewidths=1.5, alpha=0.5, label='KDE')
plt.title('Transportation Time Vulnerability vs. Average Transportation Time')
plt.xlabel('Average Transportation Time')
plt.ylabel('Transportation Time Vulnerability')
plt.grid(True)
plt.legend()
plt.show()


# In[103]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for KDE plot

# Assuming `results` is a list of dictionaries from your simulations, each containing the 'transportation_costs' and 'transportation_cost_vulnerability' data
transportation_costs = []  # Store average transportation costs for ports
vulnerabilities = []  # Store transportation cost vulnerabilities for ports

for result in results:
    # For each simulation, calculate the average transportation cost associated with each port and collect the port's vulnerability
    for port, vulnerability in result['transportation_cost_vulnerability'].items():
        # Assuming transportation_costs are stored in a dictionary with keys as edge tuples (u, v)
        port_costs = [cost for (u, v), cost in result['transportation_costs'].items() if u == port or v == port]
        if port_costs:  # Check if there are any costs associated with this port
            avg_cost = np.mean(port_costs)
            if avg_cost < 5000:  # Only consider costs lower than 20000
                transportation_costs.append(avg_cost)
                vulnerabilities.append(vulnerability)

# Plotting
plt.figure(figsize=(10, 6))
# Scatter plot
plt.scatter(transportation_costs, vulnerabilities, alpha=0.6, label='Data Points')
# KDE plot for the density distribution
sns.kdeplot(x=transportation_costs, y=vulnerabilities, levels=20, color="k", linewidths=1.5, alpha=0.5, label='KDE')
plt.title('Transportation Cost Vulnerability vs. Average Transportation Cost')
plt.xlabel('Average Transportation Cost')
plt.ylabel('Transportation Cost Vulnerability')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




