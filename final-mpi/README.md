# 2D Convolution with MPI and OpenMP

This project implements a parallel 2D convolution operation using a hybrid approach with MPI for distributed-memory parallelism and OpenMP for shared-memory parallelism.

## How to Use

### Building the Project

To build the project, simply run the `make` command in the root directory of the project. This will compile the source code and create an executable file named `conv_stride_test` in the `build` directory.

```bash
make
```

To remove the build artifacts, you can run:

```bash
make clean
```

### Installing Dependencies

The project requires an MPI implementation (like OpenMPI) and OpenMP. The `Makefile` includes a target to help install these dependencies on Debian-based (like Ubuntu) and RedHat-based (like CentOS) systems.

```bash
make install-deps
```

On HPC clusters, you will typically load the required modules:

```bash
module load mpi gcc
```

### Running the Program

The program is executed using `mpirun`. You can specify the number of processes to use with the `-np` flag.

There are two main modes of operation:

1.  **Generate and Run:** The program can generate random input and kernel matrices and then perform the convolution.
2.  **Run from File:** The program can read the input and kernel matrices from files.

### Command-line Arguments

| Flag | Argument | Description |
|---|---|---|
| `-f` | `FILE` | Path to the input feature map file. |
| `-g` | `FILE` | Path to the input kernel file. |
| `-o` | `FILE` | Path to the output file (optional). |
| `-H` | `HEIGHT` | Height of the matrix to generate. |
| `-W` | `WIDTH` | Width of the matrix to generate. |
| `-kH` | `HEIGHT` | Height of the kernel to generate. |
| `-kW` | `WIDTH` | Width of the kernel to generate. |
| `-sH` | `STRIDE` | The vertical stride (default: 1). |
| `-sW` | `STRIDE` | The horizontal stride (default: 1). |
| `-p` | `PRECISION` | Enable verify mode with a given precision (e.g., 1 for 0.1, 2 for 0.01). |
| `-s` | | Serial execution (disables OpenMP). |
| `-m` | | Enable MPI. |
| `-t` | | Time the execution in milliseconds. |
| `-T` | | Time the execution in seconds. |
| `-v` | | Enable verbose output. |
| `-h` | | Show the help message. |

### Examples

**Generate a 100x100 matrix and a 3x3 kernel, and run convolution with a 2x2 stride using 2 processes:**

```bash
mpirun -np 2 ./build/conv_stride_test -H 100 -W 100 -kH 3 -kW 3 -sH 2 -sW 2 -t
```

**Use existing files for the input and kernel, with a 3x2 stride, and save the output:**

```bash
mpirun -np 1 ./build/conv_stride_test -f input.txt -g kernel.txt -o output.txt -sH 3 -sW 2 -v
```

**Verify the output against an expected output file with a precision of 2 decimal places:**

```bash
mpirun -np 1 ./build/conv_stride_test -f input.txt -g kernel.txt -o expected_output.txt -p 2
```
