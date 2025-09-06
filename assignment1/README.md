# 2D Convolution Assignment

This project implements a fast parallel 2D convolution library using OpenMP for CITS3402/CITS5507 Assignment 1.

## Features

- **Serial and Parallel Implementations**: Both single-threaded and OpenMP parallel versions
- **"Same" Padding**: Maintains input and output dimensions
- **Flexible I/O**: Read from files or generate random matrices
- **Performance Timing**: Built-in timing functionality in milliseconds and seconds
- **Verification Mode**: Compare results with expected output using precision control
- **Command-line Interface**: Easy-to-use CLI with getopt

## Building

### Prerequisites

- GCC compiler with OpenMP support
- Make
- OpenMP development libraries

### Install Dependencies

```bash
# Install dependencies automatically
make install-deps

# Or manually:
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libgomp-devel

# macOS
brew install libomp
```

### Build

```bash
# Build the project (default)
make

# Build with debug flags
make debug

# Clean build artifacts
make clean

# Show available targets
make help
```

## Usage

### Basic Usage

```bash
# Process existing input files
./build/conv_test -f f.txt -g g.txt -o output.txt

# Generate random matrices and process
./build/conv_test -H 1000 -W 1000 -kH 3 -kW 3

# Generate and save all matrices
./build/conv_test -H 1000 -W 1000 -kH 3 -kW 3 -f f.txt -g g.txt -o output.txt
```

### Command-line Options

- `-f FILE`: Input feature map file
- `-g FILE`: Input kernel file  
- `-o FILE`: Output file (optional, prints to stdout if not specified)
- `-H HEIGHT`: Height of generated matrix (default: 1000)
- `-W WIDTH`: Width of generated matrix (default: 1000)
- `-kH HEIGHT`: Height of generated kernel (default: 3)
- `-kW WIDTH`: Width of generated kernel (default: 3)
- `-p PRECI`: Enable verify mode with precision of floating point (1 ==> 0.1)
- `-s`: Use serial implementation (default: parallel)
- `-t`: Time the execution in milliseconds
- `-T`: Time the execution in seconds
- `-v`: Verbose output
- `-h`: Show help message

### Examples

```bash
# Test with provided test cases
make test

# Compare serial vs parallel performance
make compare

# Time execution with verbose output
./build/conv_test -H 500 -W 500 -kH 3 -kW 3 -t -v

# Use serial implementation
./build/conv_test -H 1000 -W 1000 -kH 3 -kW 3 -s -t

# Verify with precision 2
./build/conv_test -f f.txt -g g.txt -o expected.txt -p 2

# Generate test matrices
./build/conv_test -H 1000 -W 1000 -kH 7 -kW 7 -Gst -f f/gen_f_1000.txt -g g/gen_g_1000.txt -o o/gen_o_1000.txt
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
- Uses OpenMP for parallel execution
- Parallelizes the convolution operation
- Supports both serial and parallel implementations

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

This will test all provided test cases with 2 threads (`-p 2`) and timing enabled (`-t`).

## Performance Analysis

For performance analysis, use the timing functionality:

```bash
# Time different matrix sizes
./build/conv_test -H 100 -W 100 -kH 3 -kW 3 -t
./build/conv_test -H 500 -W 500 -kH 3 -kW 3 -t
./build/conv_test -H 1000 -W 1000 -kH 3 -kW 3 -t

# Compare serial vs parallel
make compare

# Or manually:
./build/conv_test -H 1000 -W 1000 -kH 7 -kW 7 -s -f f/gen_f_1000.txt -g g/gen_g_1000.txt -o o/gen_o_1000.txt  # Serial
./build/conv_test -H 1000 -W 1000 -kH 7 -kW 7 -t -p 2 -f f/gen_f_1000.txt -g g/gen_g_1000.txt -o o/gen_o_1000.txt  # Parallel
```

## Verification Mode

The program supports verification mode using the `-p` option:

```bash
# Verify with precision 1 (0.1 tolerance)
./build/conv_test -f f.txt -g g.txt -o expected.txt -p 1

# Verify with precision 2 (0.01 tolerance)
./build/conv_test -f f.txt -g g.txt -o expected.txt -p 2
```

This will compare the computed result with the expected output file and report "Verify Pass!" or "Verify Failed!".

## Project Structure

```
assignment1/
├── include/
│   └── conv2d.h          # Header file with function declarations
├── src/
│   ├── main.c            # Main program with CLI
│   ├── conv2d.c          # Core convolution implementations
│   └── io.c              # Matrix I/O and generation functions
├── build/                # Build directory (created by make)
│   └── conv_test         # Executable
├── f/                    # Test input feature maps
├── g/                    # Test kernels
├── o/                    # Expected outputs
├── Makefile              # Build configuration
└── README.md             # This file
```

## Available Make Targets

- `make` or `make all`: Build the project (default)
- `make debug`: Build with debug flags
- `make clean`: Remove build artifacts
- `make test`: Run tests with provided test cases
- `make compare`: Compare serial vs parallel performance
- `make install-deps`: Install OpenMP development libraries
- `make help`: Show available targets
