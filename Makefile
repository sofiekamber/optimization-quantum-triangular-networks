main: main.cpp
	g++ main.cpp -lgmp -o code
	./code

test: test.cpp
	g++ test.cpp -o test
	./test