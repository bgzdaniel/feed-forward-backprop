feedforward: feedforward.cpp
	g++ feedforward.cpp -o feedforward -larmadillo -O3

clean:
	rm feedforward