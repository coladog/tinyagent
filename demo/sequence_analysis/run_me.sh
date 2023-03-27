cd pytorch_model_generator
python3 data_generator.py
python3 model_generator.py
cd ..

cd floating_point_based
python3 pytorch_inference.py
make
make clean
./tiny_agent_inference
cd ..

cd integer_based
python3 pytorch_inference.py
make make clean
./tiny_agent_inference
cd ..

