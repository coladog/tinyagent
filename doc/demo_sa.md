# Demo: Sequence analysis

In this demo, we build a neural network consistes of a gated recurrent unit and a multi-layer perceptron, to read in a sequence consisting 0/1.
The network will inference the number of 1 contained in this sequence and the length of this sequence.

For example, for input = [0, 1, 0, 1], the trained model should predict the length of the sequence = 4, and the number of 1 contained = 2.

***

## Generate data for training:

```
cd ./demo/sequence_analysis/pytorch_model_generator
python3 data_generator.py
```

Output:

```
[+] generated training dataset into pytorch_model_generator/data/train.pkl
```

## Train a model (GRU + MLP) via PyTorch:

```
cd ./demo/sequence_analysis/pytorch_model_generator
python3 model_generator.py
```

Output:
```
[+] Training  1/30: loss: 0.08032
[+] Training  2/30: loss: 0.05505
[+] Training  3/30: loss: 0.02690
[+] Training  4/30: loss: 0.00576
[+] Training  5/30: loss: 0.00249
[+] Training  6/30: loss: 0.00117
[+] Training  7/30: loss: 0.00066
[+] Training  8/30: loss: 0.00041
[+] Training  9/30: loss: 0.00025
[+] Training 10/30: loss: 0.00016
[+] Training 11/30: loss: 0.00011
[+] Training 12/30: loss: 0.00008
[+] Training 13/30: loss: 0.00006
[+] Training 14/30: loss: 0.00005
[+] Training 15/30: loss: 0.00004
[+] Training 16/30: loss: 0.00003
[+] Training 17/30: loss: 0.00003
[+] Training 18/30: loss: 0.00003
[+] Training 19/30: loss: 0.00002
[+] Training 20/30: loss: 0.00002
[+] Training 21/30: loss: 0.00002
[+] Training 22/30: loss: 0.00002
[+] Training 23/30: loss: 0.00002
[+] Training 24/30: loss: 0.00001
[+] Training 25/30: loss: 0.00001
[+] Training 26/30: loss: 0.00001
[+] Training 27/30: loss: 0.00001
[+] Training 28/30: loss: 0.00001
[+] Training 29/30: loss: 0.00001
[+] Training 30/30: loss: 0.00001
[+] Saved trained models into pytorch_model_generator/data/gru.model & pytorch_model_generator/data/mlp.model
```

## Inference with userspace PyTorch:

```
cd ./demo/sequence_analysis/integer_based
python3 pytorch_inference.py
```

Output:
```
[+] read param from:
[+]       ../pytorch_model_generator/data/gru.model
[+]       ../pytorch_model_generator/data/mlp.model
[+]
[+] now testing sequence:  [1, 0, 1, 1, 0, 0, 1, 0, 1]
[+]      prediction: [0.5533, 0.4490] -> number of 1: 5 | length of sequence: 9
[+] now testing sequence:  [0, 0, 1, 0, 1]
[+]      prediction: [0.4014, 0.2509] -> number of 1: 2 | length of sequence: 5
[+] now testing sequence:  [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[+]      prediction: [0.5977, 0.4978] -> number of 1: 6 | length of sequence: 10
```

**Note the result for prediction will not be exactly the same, as there's randomness in data generation and training.**

## Inference with floating-point based TinyAgent:

### Convert paramaters generated from PyTorch to TinyAgent:

```
cd ./demo/sequence_analysis/floating_point_based
python3 convert_param_from_pytorch.py
```

Output:
```
[+] converted param from PyTorch to TinyAgent:
[+]      ../pytorch_model_generator/data/gru.model -> ./data/gru.ta
[+]      ../pytorch_model_generator/data/mlp.model -> ./data/mlp.ta
```

### Compile TinyAgent:

```
cd ./demo/sequence_analysis/floating_point_based
make && make clean
```

Output:
```
gcc -Wall -g -c tiny_agent_inference.c -o tiny_agent_inference.o
gcc -Wall -g -c ../../../src/floating_point_based/tiny_agent_f.c -o tiny_agent_f.o -lm
gcc -Wall -g -c ../../../src/floating_point_based/tiny_agent_io.c -o tiny_agent_io.o
gcc -Wall -g tiny_agent_inference.o tiny_agent_f.o tiny_agent_io.o -o tiny_agent_inference -lm
rm *.o
```

### Inference with TinyAgent

```
cd ./demo/sequence_analysis/floating_point_based
./tiny_agent_inference
```

Output:
```
[+] read param from:
[+]      ./data/gru.ta
[+]      ./data/mlp.ta
[+]
[+] now testing sequence: [1, 0, 1, 1, 0, 0, 1, 0, 1]
[+]      prediction: [0.5533, 0.4490] -> number of 1: 5 | length of sequence: 9 
[+] now testing sequence: [0, 0, 1, 0, 1]
[+]      prediction: [0.4014, 0.2509] -> number of 1: 2 | length of sequence: 5 
[+] now testing sequence: [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[+]      prediction: [0.5977, 0.4978] -> number of 1: 6 | length of sequence: 10
```

**Note the result for prediction will not be exactly the same, as there's randomness in data generation and training.**

## Inference with integer-point based TinyAgent:

### Convert paramaters generated from PyTorch to TinyAgent:

```
cd ./demo/sequence_analysis/integer_based
python3 convert_param_from_pytorch.py
```

Output:
```
[+] converted param from PyTorch to integer-based TinyAgent:
[+]      ../pytorch_model_generator/data/gru.model -> ./data/gru.ta
[+]      ../pytorch_model_generator/data/mlp.model -> ./data/mlp.ta
```

### Compile TinyAgent:

```
cd ./demo/sequence_analysis/integer_based
make && make clean
```

Output:
```
gcc -Wall -Wno-unused -g -c tiny_agent_inference.c -o tiny_agent_inference.o
gcc -Wall -Wno-unused -g -c ../../../src/integer_based/tiny_agent.c -o tiny_agent.o
gcc -Wall -Wno-unused -g -c ../../../src/integer_based/tiny_agent_io.c -o tiny_agent_io.o
gcc -Wall -Wno-unused -g tiny_agent_inference.o tiny_agent.o tiny_agent_io.o -o tiny_agent_inference
rm *.o
```

### Inference with TinyAgent

```
cd ./demo/sequence_analysis/integer_based
./tiny_agent_inference
```

Output:
```
[+] read param from:
[+]      ./data/gru.ta
[+]      ./data/mlp.ta
[+]
[+] now testing sequence: [1, 0, 1, 1, 0, 0, 1, 0, 1]
[+]      prediction: [0.5533, 0.4488] -> number of 1: 5 | length of sequence: 9 
[+] now testing sequence: [0, 0, 1, 0, 1]
[+]      prediction: [0.4014, 0.2508] -> number of 1: 2 | length of sequence: 5 
[+] now testing sequence: [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[+]      prediction: [0.5977, 0.4978] -> number of 1: 6 | length of sequence: 10
```

**Note the result for prediction will not be exactly the same, as there's randomness in data generation and training.**

# Run full demo with script

```
cd ./demo/sequence_analysis
bash run_me.sh
```
