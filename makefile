feedforward: feedforward.cpp
	g++ feedforward.cpp -o feedforward -larmadillo -O2

clean:
	rm feedforward