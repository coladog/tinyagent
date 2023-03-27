tiny_agent_inference: tiny_agent_inference.o tiny_agent.o tiny_agent_io.o
	gcc -Wall -Wno-unused -g tiny_agent_inference.o tiny_agent.o tiny_agent_io.o -o tiny_agent_inference

tiny_agent_inference.o: tiny_agent_inference.c
	gcc -Wall -Wno-unused -g -c tiny_agent_inference.c -o tiny_agent_inference.o

tiny_agent.o: ../../../src/integer_based/tiny_agent.c ../../../src/integer_based/tiny_agent.h
	gcc -Wall -Wno-unused -g -c ../../../src/integer_based/tiny_agent.c -o tiny_agent.o

tiny_agent_io.o: ../../../src/integer_based/tiny_agent_io.c ../../../src/integer_based/tiny_agent_io.h
	gcc -Wall -Wno-unused -g -c ../../../src/integer_based/tiny_agent_io.c -o tiny_agent_io.o

clean:
	rm *.o
