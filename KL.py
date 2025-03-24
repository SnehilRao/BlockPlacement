# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:31:53 2025

@author: esneh
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def compute_gain(graph, partition_a, partition_b, node):
    """ Compute the gain of moving a node to the opposite partition. """
    internal = sum(1 for neighbor in graph.neighbors(node) if neighbor in (partition_a if node in partition_a else partition_b))
    external = sum(1 for neighbor in graph.neighbors(node) if neighbor in (partition_b if node in partition_a else partition_a))
    return external - internal

def kernighan_lin_algorithm(graph, partition_a, partition_b):
    """ Implements Kernighan-Lin Algorithm with swap visualization. """
    best_partition_a, best_partition_b = partition_a.copy(), partition_b.copy()
    locked = set()
    gains = []
    swap_pairs = []
    
    iteration = 0
    while len(locked) < len(graph.nodes):
        max_gain = float('-inf')
        best_pair = None
        
        # Find the best pair of nodes to swap
        for node_a in partition_a:
            if node_a in locked:
                continue
            for node_b in partition_b:
                if node_b in locked:
                    continue
                gain = compute_gain(graph, partition_a, partition_b, node_a) + compute_gain(graph, partition_a, partition_b, node_b) - 2 * (node_b in graph[node_a])
                if gain > max_gain:
                    max_gain = gain
                    best_pair = (node_a, node_b)
        
        if best_pair is None:
            break  # No more swaps possible
        
        # Swap the best pair and lock them
        node_a, node_b = best_pair
        partition_a.remove(node_a)
        partition_b.remove(node_b)
        partition_a.add(node_b)
        partition_b.add(node_a)
        locked.update([node_a, node_b])
        gains.append(max_gain)
        swap_pairs.append(best_pair)

        # Plot swap dynamically
        plot_swap(graph, partition_a, partition_b, node_a, node_b, iteration)
        iteration += 1
    
    # Determine the best cut found
    sum_gi = [0] * len(gains)
    max_gi = float('-inf')
    best_k = -1
    
    for i, g in enumerate(gains):
        sum_gi[i] = sum_gi[i - 1] + g if i > 0 else g
        if sum_gi[i] > max_gi:
            max_gi = sum_gi[i]
            best_k = i  # Store best iteration
    
    # Revert to best partition found
    best_partition_a, best_partition_b = partition_a.copy(), partition_b.copy()
    for i in range(best_k + 1, len(swap_pairs)):
        node_a, node_b = swap_pairs[i]
        best_partition_a.remove(node_b)
        best_partition_b.remove(node_a)
        best_partition_a.add(node_a)
        best_partition_b.add(node_b)
    
    return best_partition_a, best_partition_b, gains, sum_gi

def count_cuts(graph, partition_a, partition_b):
    """ Count number of edges between partitions. """
    return sum(1 for u, v in graph.edges if (u in partition_a and v in partition_b) or (u in partition_b and v in partition_a))

def plot_partitions(graph, initial_a, initial_b, final_a, final_b):
    """ Plot initial and final partitions side by side with separated colors. """
    pos = nx.bipartite_layout(graph, initial_a)  # Ensure red-left, blue-right
    
    plt.figure(figsize=(12, 6))
    
    # Initial Partition
    plt.subplot(1, 2, 1)
    nx.draw(graph, pos, with_labels=True, 
            node_color=['red' if n in initial_a else 'blue' for n in graph.nodes], 
            edge_color='gray')
    cut_count = count_cuts(graph, initial_a, initial_b)
    plt.title(f"Initial Partition (Cuts = {cut_count})")

    pos1 = nx.bipartite_layout(graph, final_a)  # Ensure red-left, blue-right

    # Final Partition
    plt.subplot(1, 2, 2)
    nx.draw(graph, pos1, with_labels=True, 
            node_color=['red' if n in final_a else 'blue' for n in graph.nodes], 
            edge_color='gray')
    cut_count = count_cuts(graph, final_a, final_b)
    plt.title(f"Final Partition (Cuts = {cut_count})")
    
    plt.show()

def plot_gains(gains, sum_gi):
    """ Plot individual gains and cumulative sum of gains. """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(gains) + 1), gains, marker='o', linestyle='-', label='g_i (Individual Gains)')
    plt.plot(range(1, len(sum_gi) + 1), sum_gi, marker='s', linestyle='-', label='Sum of g_i (Cumulative Gains)')
    plt.xlabel("Iteration")
    plt.ylabel("Gain Value")
    plt.title("Gain Progression in Kernighan-Lin Algorithm")
    plt.legend()
    plt.grid()
    plt.show()

def plot_swap(graph, partition_a, partition_b, node_a, node_b, iteration):
    """ Visualize swaps dynamically at each iteration. """
    pos = nx.bipartite_layout(graph, partition_a)  # Ensure red-left, blue-right
    
    plt.figure(figsize=(6, 5))
    nx.draw(graph, pos, with_labels=True, 
            node_color=['red' if n in partition_a else 'blue' for n in graph.nodes], 
            edge_color='gray')
    
    # Highlight swapped nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=[node_a, node_b], node_color='green', node_size=500)
    plt.title(f"Swap Iteration {iteration}: {node_a} ⇄ {node_b}")
    plt.show()

# Example usage
def run_example():
    """ Generate a random graph and run Kernighan-Lin algorithm. """
    np.random.seed(None)  # Different seeds for different runs
    graph = nx.erdos_renyi_graph(20, 0.5)  # Random graph with 20 vertices

    # Initial partition (random)
    nodes = list(graph.nodes)
    np.random.shuffle(nodes)
    partition_a = set(nodes[:10])
    partition_b = set(nodes[10:])

    # Store initial partitions
    initial_a, initial_b = partition_a.copy(), partition_b.copy()

    # Run the algorithm
    best_a, best_b, gains, sum_gi = kernighan_lin_algorithm(graph, partition_a, partition_b)

    # Plot the partitions side by side
    plot_partitions(graph, initial_a, initial_b, best_a, best_b)

    # Plot the gains
    plot_gains(gains, sum_gi)

# Run the example
run_example()
  
