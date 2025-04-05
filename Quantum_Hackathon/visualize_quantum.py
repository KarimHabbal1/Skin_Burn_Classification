import pennylane as qml
import matplotlib.pyplot as plt

def quantum_circuit(inputs, weights, n_qubits=4):
    # Encode classical data into quantum state
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum gates
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
        qml.RZ(weights[i + n_qubits], wires=i)
    
    # Entangle qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def visualize_quantum_circuit():
    # Create a device with 4 qubits
    dev = qml.device("default.qubit", wires=4)
    
    # Create the quantum node
    qnode = qml.QNode(quantum_circuit, dev)
    
    # Generate some dummy inputs and weights
    inputs = [0.1, 0.2, 0.3, 0.4]
    weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Draw the circuit
    fig, ax = qml.draw_mpl(qnode)(inputs, weights)
    
    # Customize the plot
    plt.title("Quantum Circuit Architecture", pad=20)
    plt.savefig('quantum_circuit.png', dpi=300, bbox_inches='tight')
    print("Quantum circuit visualization saved as 'quantum_circuit.png'")

if __name__ == '__main__':
    visualize_quantum_circuit() 