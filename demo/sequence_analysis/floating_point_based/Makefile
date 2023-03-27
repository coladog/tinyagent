tiny_agent_inference: tiny_agent_inference.o tiny_agent_f.o tiny_agent_io.o
	gcc -Wall -g tiny_agent_inference.o tiny_agent_f.o tiny_agent_io.o -o tiny_agent_inference -lm

tiny_agent_inference.o: tiny_agent_inference.c
	gcc -Wall -g -c tiny_agent_inference.c -o tiny_agent_inference.o

tiny_agent_f.o: ../../../src/floating_point_based/tiny_agent_f.c ../../../src/floating_point_based/tiny_agent_f.h
	gcc -Wall -g -c ../../../src/floating_point_based/tiny_agent_f.c -o tiny_agent_f.o -lm

tiny_agent_io.o: ../../../src/floating_point_based/tiny_agent_io.c ../../../src/floating_point_based/tiny_agent_io.h
	gcc -Wall -g -c ../../../src/floating_point_based/tiny_agent_io.c -o tiny_agent_io.o

clean:
	rm *.o
