# Motivation behind TinyAgent

TinyAgent uses a design pattern that separates inference from training.

It implements the inference function inside the system of the target and updates the parameters of the neural network outside the system operation.

Basically, such a design method can:
1. Reduce the overhead caused by neural networks, as the overhead of training is often much greater than inference.
2. Reduce the complexity of code added to the system internals while allowing the training of neural networks with powerful user state tools.
3. Since the neural network inside the system only does inference, it can be implemented with integers through quantization. This can be helpful in some extreme environments, such as the Linux kernel. Due to the need for high accuracy, it is often impossible to avoid using floating point numbers during the training phase.

