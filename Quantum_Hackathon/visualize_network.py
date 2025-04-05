import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arrow

def draw_neuron(ax, x, y, radius=0.4, color='blue'):
    circle = Circle((x, y), radius, color=color, alpha=0.6)
    ax.add_patch(circle)
    return circle

def draw_conv_layer(ax, x, y, width, height, num_filters, color='lightblue'):
    # Draw the main rectangle
    rect = Rectangle((x, y), width, height, color=color, alpha=0.6)
    ax.add_patch(rect)
    
    # Draw filter indicators
    filter_height = height / num_filters
    for i in range(num_filters):
        y_pos = y + i * filter_height
        filter_rect = Rectangle((x, y_pos), width, filter_height, 
                              color=color, alpha=0.8, fill=False)
        ax.add_patch(filter_rect)
    
    return rect

def draw_quantum_layer(ax, x, y, num_qubits, color='purple'):
    # Draw quantum circuit representation
    qubit_spacing = 1.0
    for i in range(num_qubits):
        # Qubit line
        plt.plot([x, x + 2], [y + i * qubit_spacing, y + i * qubit_spacing], 
                color=color, linewidth=2)
        # Qubit circle
        circle = Circle((x, y + i * qubit_spacing), 0.2, color=color)
        ax.add_patch(circle)
    
    # Draw CNOT gates
    for i in range(num_qubits - 1):
        plt.plot([x + 1, x + 1], [y + i * qubit_spacing, y + (i + 1) * qubit_spacing],
                color=color, linewidth=1, linestyle='--')
    
    return None

def visualize_network():
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set up the plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw input layer (3 channels for RGB)
    input_neurons = []
    for i in range(3):
        neuron = draw_neuron(ax, 1, 4 + i, color='red')
        input_neurons.append(neuron)
    
    # Draw first conv layer (16 filters)
    conv1 = draw_conv_layer(ax, 2, 3, 1, 3, 16)
    
    # Draw pooling layer
    pool1 = draw_conv_layer(ax, 3.5, 3, 0.5, 3, 16, color='green')
    
    # Draw second conv layer (32 filters)
    conv2 = draw_conv_layer(ax, 4.5, 3, 1, 3, 32)
    
    # Draw pooling layer
    pool2 = draw_conv_layer(ax, 6, 3, 0.5, 3, 32, color='green')
    
    # Draw quantum layer
    quantum = draw_quantum_layer(ax, 7, 3, 4)
    
    # Draw fully connected layers
    fc1_neurons = []
    for i in range(5):  # Representing 128 neurons
        neuron = draw_neuron(ax, 8.5, 3 + i * 0.5, radius=0.2, color='orange')
        fc1_neurons.append(neuron)
    
    # Draw output layer (3 classes)
    output_neurons = []
    for i in range(3):
        neuron = draw_neuron(ax, 9.5, 4 + i, color='red')
        output_neurons.append(neuron)
    
    # Add connections between layers
    # Note: In reality, these would be much more complex, this is a simplified representation
    for i in range(3):
        for j in range(16):
            plt.plot([1.4, 2], [4 + i, 3 + j/5], color='gray', alpha=0.3)
    
    # Add labels
    plt.text(1, 6, 'Input\n(RGB)', ha='center')
    plt.text(2.5, 6, 'Conv1\n(16 filters)', ha='center')
    plt.text(4, 6, 'Pool1', ha='center')
    plt.text(5, 6, 'Conv2\n(32 filters)', ha='center')
    plt.text(6.5, 6, 'Pool2', ha='center')
    plt.text(7.5, 6, 'Quantum\nCircuit', ha='center')
    plt.text(8.5, 6, 'FC1\n(128 neurons)', ha='center')
    plt.text(9.5, 6, 'Output\n(3 classes)', ha='center')
    
    plt.title('Hybrid Quantum-Classical CNN Architecture', pad=20)
    plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved as 'network_visualization.png'")

if __name__ == '__main__':
    visualize_network() 