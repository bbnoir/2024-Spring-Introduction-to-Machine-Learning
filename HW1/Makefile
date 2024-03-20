all: HW1

CXX = g++

FLAGS = --std=c++17 -Wall -pedantic -O3
LIBS = -I ../eigen

HW1: main.o
	$(CXX) $(FLAGS) -o HW1 main.o $(LIBS)

main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp $(LIBS)

clean:
	rm -f *.o HW1
