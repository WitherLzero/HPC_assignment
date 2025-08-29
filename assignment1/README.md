# 2D Convolution Assignment

This project implements a fast parallel 2D convolution library using OpenMP for CITS3402/CITS5507 Assignment 1.

## Features

- **Serial and Parallel Implementations**: Both single-threaded and OpenMP parallel versions
- **"Same" Padding**: Maintains input and output dimensions
- **Flexible I/O**: Read from files or generate random matrices
- **Performance Timing**: Built-in timing functionality
- **Command-line Interface**: Easy-to-use CLI with getopt

## Building

### Prerequisites

- GCC compiler with OpenMP support
- Make
- OpenMP development libraries

### Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libgomp-devel

# macOS
brew install libomp
```

### Build

```bash
# Build the project
make

# Build with debug flags
make debug

# Clean build artifacts
make clean
```

## Usage

### Basic Usage

```bash
# Process existing input files
./conv_test -f f.txt -g g.txt -o output.txt

# Generate random matrices and process
./conv_test -H 1000 -W 1000 -kH 3 -kW 3

# Generate and save all matrices
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -f f.txt -g g.txt -o output.txt
```

### Command-line Options

- `-f FILE`: Input feature map file
- `-g FILE`: Input kernel file  
- `-o FILE`: Output file (optional, prints to stdout if not specified)
- `-H HEIGHT`: Height of generated matrix (default: 1000)
- `-W WIDTH`: Width of generated matrix (default: 1000)
- `-kH HEIGHT`: Height of generated kernel (default: 3)
- `-kW WIDTH`: Width of generated kernel (default: 3)
- `-s`: Use serial implementation (default: parallel)
- `-t`: Time the execution
- `-v`: Verbose output
- `-h`: Show help message

### Examples

```bash
# Test with provided test cases
make test

# Run performance tests
make perf

# Compare serial vs parallel performance
make compare

# Time execution with verbose output
./conv_test -H 500 -W 500 -kH 3 -kW 3 -t -v

# Use serial implementation
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -s -t
```

## File Format

Matrices are stored in space-separated text files with the following format:

```
<height> <width>
<row1_element1> <row1_element2> ... <row1_elementN>
<row2_element1> <row2_element2> ... <row2_elementN>
...
<rowM_element1> <rowM_element2> ... <rowM_elementN>
```

Example:
```
3 4
0.884 0.915 0.259 0.937
0.189 0.448 0.337 0.033
0.122 0.169 0.316 0.111
```

## Implementation Details

### Algorithm
- Implements 2D convolution with "same" padding
- Uses zero-padding around input boundaries
- Supports arbitrary kernel sizes

### Parallelization
- Uses OpenMP with `#pragma omp parallel for collapse(2)`
- Static scheduling for load balancing
- Parallelizes the outer two loops of the convolution operation

### Memory Layout
- Row-major order for better cache locality
- Contiguous memory allocation for each row
- Efficient padding implementation

### Performance Considerations
- Cache-friendly memory access patterns
- Minimized memory allocations
- Optimized loop structure

## Testing

The project includes several test cases in the `f/`, `g/`, and `o/` directories. Run tests with:

```bash
make test
```

This will test all provided test cases and compare outputs.

## Performance Analysis

For performance analysis, use the timing functionality:

```bash
# Time different matrix sizes
./conv_test -H 100 -W 100 -kH 3 -kW 3 -t
./conv_test -H 500 -W 500 -kH 3 -kW 3 -t
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -t

# Compare serial vs parallel
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -s -t  # Serial
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -t     # Parallel
```

## Project Structure

```
assignment1/
├── include/
│   └── conv2d.h          # Header file with function declarations
├── src/
│   ├── main.c            # Main program with CLI
│   ├── conv2d.c          # Core convolution implementations
│   └── matrix_io.c       # Matrix I/O and generation functions
├── f/                    # Test input feature maps
├── g/                    # Test kernels
├── o/                    # Expected outputs
├── Makefile              # Build configuration
└── README.md             # This file
```

## Authors

[Student Name] - [Student Number]
[Student Name] - [Student Number] (if group work)

