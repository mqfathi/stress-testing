library(igraph)

# Parameters for the network
num_nodes <- 70   # Total number of nodes
m_parameter <- 3   # Number of edges to attach from a new node to existing nodes

# Generate a scale-free network
g <- barabasi.game(num_nodes, m = m_parameter, directed = FALSE)

# Assign roles to nodes based on degree
degree_dist <- degree(g)
sorted_indices <- order(degree_dist, decreasing = TRUE)
num_ports <- round(0.04 * num_nodes)  # 10% as ports
num_plants <- round(0.2 * num_nodes) # 20% as plants
num_customers <- num_nodes - num_ports - num_plants

# Create a vector for node types: 1 for customers, 2 for plants, 3 for ports
node_types <- c(rep(1, num_customers), rep(2, num_plants), rep(3, num_ports))
node_types <- node_types[sorted_indices]


# Remove inappropriate edges
edges_to_remove <- c()

for (e in E(g)) {
  ends <- ends(g, e)
  if ((node_types[ends[1]] == 1 && node_types[ends[2]] != 2) || 
      (node_types[ends[1]] == 2 && node_types[ends[2]] == 2)) {
    # Collect edge if a customer is connected to another customer or to a port
    # or if a plant is connected to another plant
    edges_to_remove <- c(edges_to_remove, e)
  }
}

# Now delete all collected edges at once
g <- delete_edges(g, edges_to_remove)


# Add edges to ensure connectivity according to the rules
for (i in 1:num_nodes) {
  if (node_types[i] == 1) {
    # Ensure each customer is connected to at least one plant
    if (length(neighbors(g, i, mode = "out")) == 0) {
      potential_plants <- which(node_types == 2)
      g <- add_edges(g, c(i, sample(potential_plants, 1)))
    }
  } 
}

for (i in which(node_types == 2)) {
  connected_ports <- neighbors(g, i, mode = "out")[node_types[neighbors(g, i, mode = "out")] == 3]
  if (length(connected_ports) == 0) {
    potential_ports <- which(node_types == 3)
    g <- add_edges(g, c(i, sample(potential_ports, 1)))
  }
}


degree_distribution <- degree(g)

# Find the top 3 nodes with the highest degree
top_nodes <- order(degree_distribution, decreasing = TRUE)[1:3]

# Create a color vector for the nodes
node_colors <- rep("white", vcount(g))
node_colors[top_nodes] <- "black"


# Plot the graph
plot(g, vertex.color = node_colors, vertex.size = 5, vertex.label = NA,edge.arrow.size = 0.5)