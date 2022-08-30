main: main.cpp
	g++ main.cpp -lgmp -o code

run: main.cpp
	g++ main.cpp -lgmp -o code
	./code

test: test.cpp
	g++ test.cpp -lgmp -o test
	./test