#!/bin/bash

echo [+] generating dataset and training model with PyTorch.	

cd pytorch_model_generator
python3 data_generator.py
python3 model_generator.py
cd ..

echo [+] testing floating-point-based TinyAgent.

cd floating_point_based
python3 pytorch_inference.py
make
make clean
./tiny_agent_inference
cd ..

echo [+] testing integer-based TinyAgent.

cd integer_based
python3 pytorch_inference.py
make 
make clean
./tiny_agent_inference
cd ..

