# Makefile for 2D Convolution Assignment
# CITS3402/CITS5507 Assignment 1

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -fopenmp
DEBUG_FLAGS = -Wall -Wextra -std=c11 -g -fopenmp -DDEBUG

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/conv2d.c $(SRC_DIR)/io.c
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BUILD_DIR)/conv_test

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

# Test with provided test cases
test: $(TARGET)
	@echo "Testing with provided test cases..."
	@echo "Test 0:"
	./$(TARGET) -f f/f0.txt -g g/g0.txt -o o/t0.txt -t
	@echo "Test 1:"
	./$(TARGET) -f f/f1.txt -g g/g1.txt -o o/t1.txt -t
	@echo "Test 2:"
	./$(TARGET) -f f/f2.txt -g g/g2.txt -o o/t2.txt -t
	@echo "Test 3:"
	./$(TARGET) -f f/f3.txt -g g/g3.txt -o o/t3.txt -t

# Performance test
perf: $(TARGET)
	@echo "Running performance tests..."
	@echo "Small matrix (100x100):"
	./$(TARGET) -H 100 -W 100 -t -kH 3 -kW 3
	@echo "Medium matrix (500x500):"
	./$(TARGET) -H 500 -W 500 -t -kH 3 -kW 3
	@echo "Large matrix (1000x1000):"
	./$(TARGET) -H 1000 -W 1000 -t -kH 3 -kW 3

# Compare serial vs parallel 
compare: $(TARGET)
	@echo "Comparing serial vs parallel performance..."
	@echo "Serial implementation (1000x1000):"
	./$(TARGET) -Gst -H 5000 -W 5000 -kH 3 -kW 3 -f f/gen_f_1000.txt -g g/gen_g_1000.txt -o o/gen_o_1000.txt
	@echo "Parallel implementation (1000x1000):"
	./$(TARGET) -t -p 2 -f f/gen_f_1000.txt -g g/gen_g_1000.txt -o o/gen_o_1000.txt

# Install dependencies (for systems that need it)
install-deps:
	@echo "Installing OpenMP development libraries..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y libomp-dev; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y libgomp-devel; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install libomp; \
	else \
		echo "Please install OpenMP development libraries for your system"; \
	fi

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Run tests with provided test cases"
	@echo "  perf       - Run performance tests"
	@echo "  compare    - Compare serial vs parallel performance"
	@echo "  install-deps - Install OpenMP development libraries"
	@echo "  help       - Show this help message"

# Phony targets
.PHONY: all debug clean test perf compare install-deps help
