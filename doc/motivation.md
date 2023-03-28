# Motivation behind TinyAgent

TinyAgent uses a design pattern that separates inference from training.
Specifically:
1. The neural network is embedded within a system to infer events. For example, predicts CPU utilization for a future period or evaluates a candidate action.
2. Export the data collected during runtime and update the neural network outside the system. For example, send it to another machine for training using PyTorch/Tensorflow, or train it when it is not needed to run the target system.
3. Load the updated neural network parameters back inside the system.

Such a design method can:
1. Reduce the overhead caused by neural networks, as the overhead of training is often much greater than inference.
2. Reduce the complexity of code added to the system codebase (think about how to compile a neural network engine into Linux kernel) while allowing the training of neural networks with powerful user state tools.
3. Since the neural network inside the system only does inference, it can be implemented with integers through quantization. This can be helpful in some extreme environments, such as the Linux kernel. Due to the need for high accuracy, it is often impossible to avoid using floating point numbers during the training phase.
