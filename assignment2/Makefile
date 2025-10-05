# Makefile for 2D Convolution with Stride Assignment
# CITS3402/CITS5507 Assignment 2 - MPI + OpenMP Implementation

# Compiler and flags
CC = mpicc
CFLAGS = -Wall -Wextra -std=c11 -O3 -fopenmp
DEBUG_FLAGS = -Wall -Wextra -std=c11 -g -fopenmp -DDEBUG

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/conv2d_mpi.c $(SRC_DIR)/io_mpi.c $(SRC_DIR)/core.c
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)


# Target executable
TARGET = $(BUILD_DIR)/conv_stride_test

# Default target
all: $(TARGET)


# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Build main executable
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $(OBJECTS) -lm -o $@

# Debug build
debug: CFLAGS = $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Test with provided test cases (stride examples)
test: $(TARGET)
	@echo "Testing Assignment 2 with stride functionality..."
	@echo "Test basic stride functionality:"
	mpirun -np 1 ./$(TARGET) -f f/f1.txt -g g/g1.txt -sH 1 -sW 1 -t -v
	@echo "Test stride 2x2:"
	mpirun -np 2 ./$(TARGET) -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -t
	@echo "Test stride 3x2:"
	mpirun -np 2 ./$(TARGET) -H 200 -W 200 -kH 5 -kW 5 -sH 3 -sW 2 -t

# Performance comparison across different configurations
compare: $(TARGET)
	@echo "Comparing different parallelization strategies..."
	@echo "Sequential (stride 1x1):"
	mpirun -np 1 ./$(TARGET) -H 1000 -W 1000 -kH 3 -kW 3 -sH 1 -sW 1 -t -s
	@echo "Sequential (stride 2x2):"
	mpirun -np 1 ./$(TARGET) -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -t -s
	@echo "MPI only (2 processes, stride 2x2):"
	mpirun -np 2 ./$(TARGET) -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -t -m
	@echo "OpenMP only (stride 2x2):"
	mpirun -np 1 ./$(TARGET) -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -t -p
	@echo "Hybrid MPI+OpenMP (2 processes, stride 2x2):"
	mpirun -np 2 ./$(TARGET) -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -t

# Stress test for HPC performance analysis
stress: $(TARGET)
	@echo "Stress testing large matrices with different stride values..."
	@echo "Large matrix (4000x4000) stride 1x1:"
	mpirun -np 4 ./$(TARGET) -H 4000 -W 4000 -kH 5 -kW 5 -sH 1 -sW 1 -t
	@echo "Large matrix (4000x4000) stride 2x2:"
	mpirun -np 4 ./$(TARGET) -H 4000 -W 4000 -kH 5 -kW 5 -sH 2 -sW 2 -t
	@echo "Large matrix (4000x4000) stride 3x3:"
	mpirun -np 4 ./$(TARGET) -H 4000 -W 4000 -kH 5 -kW 5 -sH 3 -sW 3 -t

# Install MPI dependencies (for systems that need it)
install-deps:
	@echo "Installing MPI and OpenMP development libraries..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y libopenmpi-dev libomp-dev; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y openmpi-devel libgomp-devel; \
	elif command -v module >/dev/null 2>&1; then \
		echo "On HPC systems, load appropriate modules: module load mpi gcc"; \
	else \
		echo "Please install MPI and OpenMP development libraries for your system"; \
	fi

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Run tests with stride functionality"
	@echo "  compare    - Compare different parallelization strategies"
	@echo "  stress     - Stress test for HPC performance analysis"
	@echo "  install-deps - Install MPI and OpenMP development libraries"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  mpirun -np 4 ./build/conv_stride_test -H 1000 -W 1000 -kH 3 -kW 3 -sH 2 -sW 2 -t"

# Phony targets
.PHONY: all debug clean test compare stress install-deps help
