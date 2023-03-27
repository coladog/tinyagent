![](assets/logo.png)
# TinyAgent: Easily applying neural network inside systems.

TinyAgent is a tool to help experiment with machine learning for systems.
TinyAgent generates neural networks in systems (Linux kernel, database kernel, middleware, etc. ) with the following design pattern:
1. The neural network is embedded within a system to inference events. For example, predicts CPU utilization for a future period or evaluates an candidate action.
2. Export the data collected during runtime and update the neural network outside the system. For example, send it to another machine for training, or train it when it is not needed to run the target system.
3. Load the updated neural network parameters back inside the system.

See [motivation](./doc/motivation.md) behind it.

As of now, TinyAgent has the following features:
1. Supports gated recurrent unit (GRU, a variant of RNN) and fully connected layers. Supports ReLU, Tanh and Sigmoid activation functions. The currently supported models are sufficient to handle fixed-length features as well as indeterminate-length sequences.
2. C-based implementation (compatible with compilation into Linux kernel).
3. Contains only four source files, ~ 500 lines of code.
4. **Designed to read the model generated by PyTorch training and make the same inference**.
5. Supports fully integer-based inference and floating-point-based inference.
6. **No computational dependencies on other libraries**.  Floating-point-based inference requires math.h provided in userspace C library.

## Usage

Create a GRU (Gated Recurrent Unit, an RNN model) unit and load the parameters:
```c
struct gru_cell cell;
init_gru_cell(&cell, GRU_HIDDEN_SIZE, GRU_INPUT_SIZE);
cell.read(&cell, GRU_LOAD_PATH);
```
Clear the information of the hidden vector of the GRU cell:
```c
cell.clear(&cell);
```
Fusing data into the hidden vector of GRU cell:
```c
cell.inference(&cell, &DATA);
```
Delete the GRU cell:
```c
cell.free(&cell);
```

Define a multi-layer perceptron with one hidden layer and sigmoid activation function, and load the parameters:
```c
struct fc_layer layer1, layer2;
init_fc_layer(&layer1, FC1_INPUT_SIZE, FC2_INPUT_SIZE, SIGMOID);
init_fc_layer(&layer2, FC2_INPUT_SIZE, PREDICTION_SIZE, NONE);

struct mlp nn;
init_mlp(&nn);
nn.add_layer(&nn, &layer1);
nn.add_layer(&nn, &layer2);
nn.read(&nn, MLP_LOAD_PATH);
```
Inference data with the multi-layer perceptron:
```c
nn.inference(&nn, cell.hidden);
```
Delete the multi-layer perceptron:
```c
nn.free(&nn);
```

## Integer-based computing

The integer-based inference is done through quantization technique.
See [how](./doc/quantization.md) we design and implement it.

## Demo

We provide a [demo](./doc/demo_sa.md) to show how to train a neural network using PyTorch and use TinyAgent to read the trained network and make inference based on floating-point/integer.