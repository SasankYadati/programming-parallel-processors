NVCC = nvcc

all: vector_addition
	./vector_addition

vector_addition: main.cu vector_addition.cu
	$(NVCC) -O3 main.cu vector_addition.cu -o vector_addition

clean:
	rm -f vector_addition