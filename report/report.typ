#set document(
  title: "CITS3402/CITS5507 Assignment 1 Report",
  author: "Your Name",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "Linux Libertine",  // Professional serif font, built-in to Typst
  size: 11pt,
  lang: "en",
)

#set heading(numbering: "1.1.")

#set par(justify: true, leading: 0.6em)


#let codefence(body)={
  block(
  stroke: 0.5pt ,
  inset: 1em,
  width: 100%,
  breakable: true,
  body
)
}

// Title page
#align(center + top)[
  #v(15%)  // Position at 1/4 of the page from top
  // UWA Logo
  #image("uwa-logo.png", width: 60%)
]

#align(center)[
  #v(10%)  // Adjust this value to move content up or down
  
  #text(size: 20pt, weight: "bold")[
    CITS3402/CITS5507 Assignment 1
  ]
  
  #text(size: 16pt, style: "italic")[
    Semester 2, 2025
  ]


  #v(2em)
  
  #text(size: 16pt, weight: "bold")[
    Fast Parallel 2D Discrete Convolution Implementation
  ]
  

  
  #v(6em)
  
  #rect(
    width: 65%,
    stroke: (thickness: 1pt, dash: "dashed"),
    inset: 1.2em,
  )[
    #text(size: 14pt, weight: "bold")[
      *Submitted in group:*
    ]
    
    #v(0.5em)
    
    #text(size: 12pt)[
      Your Name, Your Student Number \
      Teammate Name, Teammate Student Number
    ]
  ]


]

#pagebreak()

// Table of contents
#outline()

#pagebreak()


= Introduction

== Problem Statement and Objectives

2D convolution is a fundamental mathematical operation extensively used in signal processing, computer vision, and machine learning applications. In convolutional neural networks (CNNs), hundreds of thousands of convolution operations are performed during inference on high-resolution images, making computational efficiency critical for practical applications.

The discrete 2D convolution of an input feature map $f$ and kernel $g$ is mathematically defined as:

$ (f * g)[n,m] = sum_(i=-M)^M sum_(j=-N)^N f[n+i, m+j] dot g[i,j] $

#figure(
  image("figures/convolution_example.png", width: 80%),
  caption: [Example of 2D convolution operation ]
) <fig:convolution-example>




This assignment focuses on developing a high-performance parallel implementation of 2D convolution with "same" padding using OpenMP. The primary objectives are:

- Implement both serial and parallel versions of 2D convolution
- Achieve significant speedup through effective parallelization
- Analyze performance characteristics and scalability
- Evaluate memory layout and cache optimization strategies

== Testing Environment

Performance analysis was conducted using a two-stage approach to ensure comprehensive evaluation and result validation.

*Primary Testing Environment - Kaya HPC Cluster:*
All primary performance analysis and scalability testing were conducted on the Kaya high-performance computing cluster, which provides:
- Multi-core Intel processors with consistent performance characteristics
- Hierarchical memory system with multiple cache levels (L1, L2, L3)
- OpenMP-enabled GCC compiler environment
- Controlled computational resources for reliable benchmarking
- Support for multi-threading analysis up to 16+ cores

*Development and Validation Environment:*
Initial development, debugging, and correctness verification were performed on local development machines to enable rapid iteration and testing. Local testing ensured code correctness across different system configurations before deployment to Kaya for performance analysis.


== Report Overview


= Implementation

== Algorithm Overview

2D convolution slides a kernel over an input matrix, computing weighted sums at each position. A key challenge in 2D convolution is managing output dimensions. Without modification, applying a kernel of size $k_H times k_W$ to an input of size $H times W$ results in an output of size $(H-k_H+1) times (W-k_W+1)$, which is smaller than the input.

For many applications like neural networks, it's desirable to maintain the same output dimensions as the input. We address this using "same" padding—a technique where zeros are added around the input matrix boundaries before applying the convolution. This ensures that the output dimensions exactly match the input dimensions.

Our implementation calculates the necessary padding as follows:
- Pad top = $(k_H - 1) / 2$
- Pad left = $(k_W - 1) / 2$
- Pad bottom = $k_H - 1 - "pad"\_"top"$
- Pad right = $k_W - 1 - "pad"\_"left"$

This asymmetric padding approach handles both odd and even kernel sizes correctly. For odd-sized kernels (e.g., 3×3), padding is equal on all sides. For even-sized kernels (e.g., 4×4), padding is asymmetric to ensure output dimensions match input dimensions precisely.

#figure(
  image("figures/convolution-padding-example.png", width: 80%),
  caption: [Example of 2D convolution with the "same" padding ]
) <fig:convolution-padding-example>

To implement "same" padding, we developed the `generate_padded_matrix` function which creates a padded version of the input matrix:


#codefence(```c
void generate_padded_matrix(float **input, int height, int width,
                           int kernel_height, int kernel_width,
                           float ***padded, int *padded_height,
                           int *padded_width) {
    // Calculate padding on each side
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;
    
    // Calculate padded dimensions
    *padded_height = height + pad_top + pad_bottom;
    *padded_width = width + pad_left + pad_right;
    
    // Allocate and initialize padded matrix with zeros
    *padded = allocate_matrix(*padded_height, *padded_width);
    for (int i = 0; i < *padded_height; i++) {
        for (int j = 0; j < *padded_width; j++) {
            (*padded)[i][j] = 0.0f;
        }
    }
    
    // Copy original data to the center of padded matrix
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            (*padded)[i + pad_top][j + pad_left] = input[i][j];
        }
    }
}```)

This function first calculates the required padding on each side based on the kernel dimensions. Then it allocates a new matrix with the appropriate padded dimensions and initializes all elements to zero. Finally, it copies the original input data to the center of the padded matrix, surrounded by the zero padding. This ensures that when convolution is applied to the padded matrix, the output dimensions will exactly match the original input dimensions.


== Serial Implementation

Our serial implementation provides the foundation for the parallel version and serves as a baseline for performance comparison. The process consists of two main steps:

1. *Padding the input matrix*: As established in the prior discussion, zeros are added around the input matrix according to the kernel dimensions. 
2. *Performing the convolution*: Four nested loops iterate through each output position and apply the kernel.

The core convolution algorithm is implemented as follows:
- *Outer loops (i, j)*: Iterate through each position in the output matrix
- *Inner loops (ki, kj)*: Apply the kernel at each output position by iterating through kernel elements
- *Accumulation*: Compute the weighted sum for each output element

#codefence(```c
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW,
                   float **output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform convolution producing an output of size out_H x out_W
    for (int i = 0; i < out_H; i++) {        // Output row iteration
        for (int j = 0; j < out_W; j++) {    // Output column iteration
            float sum = 0.0f;                // Initialize accumulator

            // Apply kernel at position (i,j)
            for (int ki = 0; ki < kH; ki++) {     // Kernel row iteration
                for (int kj = 0; kj < kW; kj++) { // Kernel column iteration
                    sum += f[i + ki][j + kj] * g[ki][kj];
                }
            }

            output[i][j] = sum;              // Store computed value
        }
    }
}"```)

=== Computational Complexity Analysis

1. *The time complexity* of this implementation is $O(H times W times k_H times k_W)$, where:
- $H times W$ represents the number of output elements to compute
- $k_H times k_W$ represents the operations per output element

For a typical scenario with input size $1000 times 1000$ and kernel size $3 times 3$, this results in approximately $9 times 10^9$ floating-point operations, making efficient implementation crucial for practical performance.

2. *The space complexity* is $O((H + k_H - 1) times (W + k_W - 1))$ for the padded input matrix, plus $O(H times W)$ for the output matrix.

=== Memory Access Pattern Analysis

The serial implementation exhibits the following memory access characteristics:

1. *Sequential access for output*: The outer loops iterate through the output matrix in row-major order, providing excellent spatial locality for output writes
2. *Stride access for input*: For each output position, the algorithm accesses a $k_H times k_W$ region of the input matrix, creating a sliding window pattern
3. *Repeated kernel access*: The kernel matrix is accessed repeatedly for each output position, making it an ideal candidate for cache retention

This access pattern is generally cache-friendly for the output matrix but can lead to cache misses for large input matrices when the working set exceeds cache capacity.

== Supporting Infrastructure

To enable comprehensive testing and performance analysis, our implementation provides optimized infrastructure for matrix operations and user interaction:

=== Optimized Random Matrix Generation

Our implementation features a high-performance random matrix generation system designed for large-scale testing:

#codefence(```c
// Fast random number generator using xorshift algorithm
static inline unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float **generate_random_matrix(int rows, int cols, float min_val,
                               float max_val) {
    float **matrix = allocate_matrix(rows, cols);
    
    // Pre-calculate range for efficiency
    const float range = max_val - min_val;
    const float inv_max = 1.0f / (float)UINT_MAX;
    const float scale = range * inv_max;

    // Use OpenMP for parallel generation
    #pragma omp parallel
    {
        // Each thread gets its own random state
        unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 12345);
        
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Generate random float using fast xorshift
                unsigned int rand_int = xorshift32(&seed);
                matrix[i][j] = ((float)rand_int * scale) + min_val;
            }
        }
    }

    return matrix;
}"```)


This implementation provides several performance advantages:
- *Fast xorshift algorithm*: Significantly faster than standard `rand()` function
- *Parallel generation*: Uses OpenMP to generate matrix elements concurrently
- *Thread-safe seeding*: Each thread maintains its own random state
- *Optimized scaling*: Pre-calculates scaling factors to avoid repeated computation

=== File I/O Operations

The file format follows the assignment specification with optimized I/O operations:

Reading matrices from files:

#codefence(```c
int read_matrix_from_file(const char *filename, float ***matrix, int *rows,
                          int *cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error: Cannot open file");
        return -1;
    }

    // Read dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        perror("Error: Cannot read matrix dimensions");
        fclose(file);
        return -1;
    }

    // Allocate matrix and read data
    *matrix = allocate_matrix(*rows, *cols);
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fscanf(file, "%f", &(*matrix)[i][j]) != 1) {
                perror("Error: Cannot read matrix element");
                free_matrix(*matrix, *rows);
                fclose(file);
                return -1;
            }
        }
    }

    fclose(file);
    return 0;
}
```)

Writing matrices maintains the same format with proper error handling and resource cleanup.

=== Command-Line Interface

A comprehensive command-line interface provides flexible testing capabilities:

```bash
Usage: ./conv_test [OPTIONS]
Options:
  -f FILE     Input feature map file
  -g FILE     Input kernel file
  -o FILE     Output file (optional)
  -H HEIGHT   Height of generated matrix (default: 1000)
  -W WIDTH    Width of generated matrix (default: 1000)
  -kH HEIGHT  Height of generated kernel (default: 3)
  -kW WIDTH   Width of generated kernel (default: 3)
  -p  PRECI   Enable verify mode with floating point precision
  -s          Use serial implementation (default: parallel)
  -t          Time execution in milliseconds
  -T          Time execution in seconds
  -v          Verbose output
  -h          Show help message
```

This interface enables various testing scenarios:
- Performance benchmarking with custom matrix dimensions
- Correctness verification against reference outputs
- Flexibility between serial and parallel implementations
- Detailed timing analysis with multiple precision options