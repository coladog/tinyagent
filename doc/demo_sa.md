# Demo: Sequence analysis

In this example, we build a neural network consistes of a gated recurrent unit and a multi-layer perceptron, to read in a sequence consisting 0/1.
The network will inference the number of 1 contained in this sequence and the length of this sequence.

First, the demo will generate two pythorch models. The ./demo/pythorch_model_generator directory includes 3 .py file to generate a dataset and then gernerate models to train the dataset.
1. data_generator.py: this file will generate 10,000 sequences which are randomly consist of 0's and 1's. The length of each sequence is also randomly generated and lies between 1 and 20. Each sequence has a label and consists of a 2-tuple, which is (length of 1/length of sequence, length of sequence/max sequence length). In this demo, max sequence length is set to 20 (in util.py). For exameple, [[1], [1], [1]] is a random sequence. Then its label is (1.0, 0.15). The 10,000 sequences and tags will be stored under data/train.pkl after running the python code.
2. model_generator.py: this file will generate a gru and mlp model based on pyTorch. The architecture of gru and mlp are defined in util.py. In this demo, gru only has one layer with 1 input and 4 hidden. The hidden state output from gru is the input to mlp. The mlp has the architecture with 4-4-2. The output is the 2-tuple label. Thus, this file read the data from data/train.pkl and use the data to train gru and mlp. The parameters of these two model will be stored in gru.model and mlp.model.

Next, floating point calculation or integer calculation can be choosed. Both of these are used in the same way. 

Floating point based is under floating_point_based directory and integer based is under integer_based directory.
Here is for floating based.
1. convert_param_from_pytorch.py: this file convert the pythorch model paramters to the parameter that can be used in TinyAgent(.ta file). 
2. tiny_agent_inference.c: this code read the parameters stored in .ta file and use the methond in TinyAgent to construct a model in code which can do the same caculation with model contructed by Pytorch. For example, the model we created in this demo is one layer gru and 4-4-1 mlp, so use init_gru_ceil(args) to create one layer gru and init_fc_layer(args) to get the 4-4-1 fully connected layers. Then the created model will be used to test. In this demo, we test 3 times. To run this file, need to make and run ./tiny_agent_inference.
We also use the orignial pytorch model to test.
3. pytorch_inference.py: is still use the pytorch model to test.

We could compare the testing results generated form step 2 and step 3, the results are the same.

Here is for integer based, which is almost same with floating based.
1. convert_param_from_pytorch.py: this file convert the pythorch model paramters to the parameter that can be used in TinyAgent(.ta file). 
2. generate_activation_table.py: this file is for activation fuction. The original activation function in the neural network was calculated using floating point numbers, here it is transformed into a tabular form without direct calculation. The table is queried by reading and writing from a file under \data. In this demo, the original sigmoid will be used through sigmoid_table_input.ta and sigmoid_table_output.ta; original tanh function will be used through tanh_table_input.ta and tanh_table_output.ta.

The following 2 steps are the same with floating based.

3. tiny_agent_inference.c: this code read the parameters stored in .ta file and use the methond in TinyAgent to construct a model in code which can do the same caculation with model contructed by Pytorch. For example, the model we created in this demo is one layer gru and 4-4-1 mlp, so use init_gru_ceil(args) to create one layer gru and init_fc_layer(args) to get the 4-4-1 fully connected layers. Then the created model will be used to test. In this demo, we test 3 times. To run this file, need to make and run ./tiny_agent_inference.
We also use the orignial pytorch model to test.
4. pytorch_inference.py: is still use the pytorch model to test.

We could still compare the testing results generated form step 2 and step 3, the results are the same.
