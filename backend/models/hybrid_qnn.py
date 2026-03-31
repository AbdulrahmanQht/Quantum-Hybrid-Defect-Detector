"""
Hybrid Quantum-Classical Neural Network (QNN) for Industrial Defect Detection.

Architecture:
    1. Classical Feature Extractor (Conv2d layers) - compresses 1x256x256 → n_qubits features
    2. Variational Quantum Circuit (PennyLane)     - quantum-enhanced feature processing
    3. Classical Classifier Head (Linear)           - maps quantum output → 6 defect classes

The VQC uses angle encoding to map classical features onto qubit rotations,
then applies parameterized gates (trainable weights) with entanglement.
PennyLane's TorchLayer makes the circuit behave like a standard nn.Module,
so the whole model trains end-to-end with normal PyTorch backprop.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.

    Args:
        num_classes:    Number of output classes (default 6 for defect types).
        n_qubits:       Number of qubits in the VQC (default 4).
        q_depth:        Number of variational layers in the VQC (default 2).
        q_device_name:  PennyLane device backend. Use "default.qubit" for CPU
                        simulation or "lightning.gpu" for cuQuantum acceleration.
    """

    def __init__(self, num_classes=6, n_qubits=4, q_depth=2, q_device_name="default.qubit"):
        super().__init__()

        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.num_classes = num_classes

        # ─── Classical Feature Extractor ───
        # Compresses 1x256x256 grayscale image down to `n_qubits` features.
        # These features become the input to the quantum circuit.
        self.feature_extractor = nn.Sequential(
            # Block 1: 1x256x256 → 16x128x128
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 16x128x128 → 32x64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 32x64x64 → 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 64x32x32 → 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Flatten and compress to n_qubits values
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            # Tanh to bound values to [-1, 1] → scaled to [-π, π] for angle encoding
            nn.Tanh(),
        )

        # ─── Quantum Layer (VQC) ───
        self.q_device = qml.device(q_device_name, wires=n_qubits)
        self.q_layer = self._build_quantum_layer()

        # ─── Classical Classifier Head ───
        # Takes the n_qubits measurement outputs from VQC → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def _build_quantum_layer(self):
        """
        Builds the Variational Quantum Circuit and wraps it as a TorchLayer.

        Circuit structure per variational layer:
            1. Angle encoding: RY(feature * π) on each qubit
               (only on first layer; re-encoding is optional)
            2. Variational rotations: RY(θ) + RZ(φ) on each qubit (trainable)
            3. Entanglement: Ring of CNOTs connecting adjacent qubits

        Measurement: ⟨PauliZ⟩ on each qubit → returns n_qubits float values in [-1, 1]
        """
        n_qubits = self.n_qubits
        q_depth = self.q_depth

        @qml.qnode(self.q_device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding: map classical features → qubit rotations
            # inputs are in [-1, 1] from Tanh, scale to [-π, π]
            for i in range(n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)

            # Variational layers
            for layer in range(q_depth):
                # Parameterized single-qubit rotations
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)

                # Entanglement: ring of CNOTs
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # Measure each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Define the shape of trainable weights: (q_depth, n_qubits, 2) for RY+RZ
        weight_shapes = {"weights": (q_depth, n_qubits, 2)}

        # Wrap as a PyTorch module — this is what makes backprop work seamlessly
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def to(self, *args, **kwargs):
        """
        Override .to() so the quantum layer ALWAYS stays on CPU.
        PennyLane's default.qubit simulator cannot run on CUDA or MPS.
        Only the classical layers (feature_extractor, classifier) move to GPU.
        """
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        # q_layer stays on CPU — do NOT move it
        return self

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 256, 256)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Remember input device (mps, cuda, cpu) for later
        input_device = x.device

        # Classical feature extraction on GPU: (B, 1, 256, 256) → (B, n_qubits)
        features = self.feature_extractor(x)

        # PennyLane's quantum simulator only runs on CPU.
        # q_layer is permanently on CPU (see .to() override above).
        # Move features to CPU, run quantum circuit, move output back.
        features_cpu = features.to("cpu")
        q_results = []
        for i in range(features_cpu.shape[0]):
            q_results.append(self.q_layer(features_cpu[i]))
        q_out = torch.stack(q_results).to(input_device)

        # Classical classification on GPU: (B, n_qubits) → (B, num_classes)
        logits = self.classifier(q_out)

        return logits
