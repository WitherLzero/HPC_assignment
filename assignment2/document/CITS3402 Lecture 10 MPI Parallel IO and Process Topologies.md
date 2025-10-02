# CITS3402 Lecture 10: MPI Parallel I/O and Process Topologies
## Comprehensive Study Notes for High Performance Computing

**Lecture Date**: October 1, 2025  
**Topics**: Parallel File I/O, Independent/Collective I/O, Process Topologies  
**Slide Reference**: lecture_10.pdf  
**Examinable**: Yes - All content covered

---

## **Key Insights Box**
**Main Learning Outcomes:**
- Understand parallel file I/O strategies and their performance implications
- Master independent and collective I/O operations in MPI
- Learn to use file views and offsets for non-contiguous data access
- Implement Cartesian and graph topologies for structured communication
- Recognize the four levels of I/O access patterns and their performance trade-offs

**Critical Concepts:**
- **File views** enable efficient non-contiguous access to shared files
- **Collective I/O** provides orders of magnitude better performance than independent I/O
- **Virtual topologies** simplify communication patterns and improve performance
- **Offsets** are essential for preventing race conditions in parallel writes

**Common Pitfalls:**
- Forgetting to use offsets leads to race conditions (all processes write to same location)
- Not calling MPI_File_close (it's collective - affects all processes)
- Confusing file modes (read-only vs write-only vs read-write)
- Assuming periodic=0 means periodic boundary conditions (it means FALSE)

---

## **PART 1: Introduction to Parallel File I/O (Slides 5-9)**

### **Why Parallel I/O Matters (Slide 6)**

**Problem Statement**:
- Supercomputers handle massive datasets
- Reading/writing data takes linear time (O(n))
- I/O can become a major bottleneck in HPC applications
- Serial I/O doesn't scale with processor count

**Solution - Parallel I/O**:
- Multiple processes read/write to files concurrently
- Requires careful coordination to avoid data corruption
- MPI provides standardized parallel I/O capabilities
- Can achieve significant speedups when done correctly

**Key Consideration**:
Supercomputers have **bespoke file systems** (custom-designed for the hardware). Examples:
- **Lustre**: Common in many HPC centers
- **GPFS**: IBM's parallel file system
- **Setonix**: Has optimized parallel file system

Understanding the file system architecture is crucial for optimal performance, but modern systems handle many optimizations automatically.

### **I/O Strategy Overview (Slide 7)**

**Basic Data Flow**:
```
Data Source → Program → Data Destination
     ↓                      ↓
   (Read)               (Write)
```

This simple model becomes complex in parallel systems where multiple processes need coordinated access.

### **Memory Hierarchy (Slide 8)**

**Storage Speed Hierarchy** (fastest → slowest):
```
CPU Registers (fastest, smallest, ~1 ns)
    ↓
L1 Cache (~1-2 ns)
    ↓
L2 Cache (~4-10 ns)
    ↓
L3 Cache (~10-20 ns)
    ↓
Main RAM/DRAM (~50-100 ns)
    ↓
SSD (Solid State Drive) (~50-150 μs)
    ↓
Hard Disk Drive (slowest, largest, ~5-10 ms)
```

**Important Notes**:
- **Cache is expensive**: Building entire systems from cache memory would be prohibitively costly
- **Hard drives are slow**: Old spinning-disk HDDs are much slower than SSDs (orders of magnitude)
- **Modern systems use SSDs**: Most current systems have replaced HDDs with faster SSDs
- **Enterprise systems**: Some still use HDDs with up to 9 platters for massive storage capacity
- **Performance gap**: The speed difference between memory and disk is enormous (millions of times)

**Why This Matters for Parallel I/O**:
- I/O operations are extremely slow compared to computation
- Optimizing I/O can have massive impact on overall performance
- Parallel I/O helps bridge the speed gap by using multiple paths

### **Hard Drive Anatomy (Slide 9)**

**Components of Traditional Hard Drives**:

```
Hard Drive Structure:
┌─────────────────────────────────────┐
│  Read/Write Heads                   │
│      ↓                               │
│  ┌─────────────┐  ← Platters       │
│  │ ╱───────╲   │    (double-sided) │
│  │ │       │   │                    │
│  │ ╲───────╱   │                    │
│  └─────────────┘                    │
│                                      │
│  Multiple platters stacked          │
│  (1-3 consumer, up to 9 enterprise) │
└─────────────────────────────────────┘

Platter Terminology:
┌──────────────────────────────────────┐
│  A = Track (concentric circle)       │
│  B = Subdivision of platter          │
│  C = Sector (subdivision of track)   │
│  D = Cluster (group of sectors)      │
│                                       │
│    ┌─────────┐                       │
│    │  ○ ○ ○  │  ← Sectors            │
│    │ ○     ○ │     within track      │
│    │  ○ ○ ○  │                       │
│    └─────────┘                       │
└──────────────────────────────────────┘
```

**Terminology**:
- **Track (A)**: One of the concentric circles on a platter
- **Subdivision (B)**: A section of the entire platter surface
- **Sector (C)**: Subdivision of a track - **minimum storage unit** for HDDs
- **Cluster (D)**: Grouping of sectors - **smallest unit file system manages**

**Historical Context**:
This level of detail was important when programmers needed to optimize for spinning-disk performance. Modern parallel file systems (like on Setonix) handle these optimizations automatically, but understanding the concepts helps explain why certain I/O patterns are faster than others.

---

## **PART 2: I/O Strategies (Slides 10-13)**

### **Non-Parallel I/O - Worst Case (Slide 10)**

**Implementation**:
```
Process 1 ─┐
Process 2 ─┼─→ Process 0 ──→ File
Process 3 ─┘   (bottleneck)

Step 1: All processes send data to P0
Step 2: P0 writes everything serially
Step 3: P0 closes file
```

**Characteristics**:
- All processes send write requests to one designated process (typically P0)
- Single process writes all data to file sequentially
- **Major bottleneck**: Communication overhead + serial write
- Gets the job done but **very slow**
- No parallelism in actual file writing

**When This Happens**:
- Using standard C I/O (fopen, fwrite) in MPI programs
- Not using MPI-specific I/O functions
- Legacy code that hasn't been updated for parallel I/O

### **Independent Parallel I/O (Slide 11)**

**Implementation**:
```
Process 0 ──→ File_0.dat
Process 1 ──→ File_1.dat
Process 2 ──→ File_2.dat
Process 3 ──→ File_3.dat

Each process writes to separate file
Fully parallel - no coordination needed
```

**Characteristics**:
- Each process writes to its own separate file
- Fully parallelized (no bottleneck)
- No synchronization required between processes
- **Drawback**: Generates many small files to manage
- Post-processing often needed to combine files
- Suitable when processes produce independent results

**Use Cases**:
- Checkpoint/restart files (each process saves its state)
- Logging or debugging output per process
- When data naturally partitions into independent pieces

### **Cooperative Parallel I/O (Slide 12)**

**Implementation**:
```
Process 0 ─┐
Process 1 ─┼─→ Shared File
Process 2 ─┤   (coordinated access)
Process 3 ─┘

All processes write to same file concurrently
Coordination handled by MPI library
```

**Characteristics**:
- All processes write to a **shared file** simultaneously
- Requires coordination to avoid data corruption
- **Only possible through MPI**: Standard file I/O cannot do this safely
- Provides best of both worlds: parallel performance + single output file
- MPI handles synchronization and coordination

**Key Advantage**: 
Combines parallelism with manageable output (one file instead of many). This is the **preferred approach** for most parallel I/O scenarios.

---

## **PART 3: MPI I/O Fundamentals (Slides 13-18)**

### **Why Use MPI for I/O? (Slide 13)**

**Advantages**:
1. **Similar to communication**: Reading/writing like send/receive operations
2. **Leverage MPI capabilities**:
   - Collective operations
   - User-defined data types
   - Non-blocking operations
   - Communicators for coordination

**What This Lecture Covers**:
- Independent I/O (each process operates independently)
- Cooperative/Collective I/O (processes coordinate)

### **File Operations Schematic (Slide 14)**

**Standard C I/O**:
```c
FILE *fp = fopen("file.dat", "r");  // Open
// ... process data ...
fclose(fp);                          // Close
```

**MPI I/O**:
```c
MPI_File fh;
MPI_File_open(comm, "file.dat", mode, info, &fh);  // Open
// ... process data ...
MPI_File_close(&fh);                                 // Close
```

**Key Similarity**: Same open → process → close pattern

**Key Difference**: MPI functions are **collective** or **independent** operations involving multiple processes

### **Rationale for Parallel File I/O (Slide 15)**

**Core Concept**: Each process describes **its part** of the file

**File View**:
- Defines what portion of file a process can see/access
- Specified using **offsets** (byte positions in file)
- Allows non-contiguous access patterns
- Enables shared access (multiple processes accessing same file)

**Key Performance Factor**:
> "Collective input/output combined with non-contiguous access is the key to the highest performance"

This combination allows the MPI library to optimize I/O operations across all processes.

---

## **PART 4: Independent I/O Operations (Slides 16-23)**

### **Types of I/O (Slide 16)**

MPI supports two fundamental types:

**1. Independent I/O**:
- Similar to point-to-point communication
- Each process operates independently
- No coordination required
- Example: `MPI_File_write()`

**2. Collective I/O**:
- Similar to collective communication
- All processes in communicator participate
- Coordination for optimization
- Example: `MPI_File_write_all()`

### **Independent I/O Workflow (Slide 17)**

**MPI I/O Steps**:
```
1. MPI_File_open()   - Open file (collective on communicator)
2. MPI_File_write()  - Write data (independent)
3. MPI_File_close()  - Close file (collective)
```

**File Modes** (similar to C I/O):
- `MPI_MODE_RDONLY`: Read only
- `MPI_MODE_WRONLY`: Write only  
- `MPI_MODE_RDWR`: Read and write

**Additional Modes**:
- `MPI_MODE_CREATE`: Create file if it doesn't exist
- Can combine modes with bitwise OR: `MPI_MODE_CREATE | MPI_MODE_WRONLY`

### **Example: Basic MPI I/O (Slide 18)**

**Complete Code Example**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, buf;
    MPI_File fh;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Initialize buffer with rank value
    buf = rank;
    
    // Open file (collective operation)
    MPI_File_open(MPI_COMM_WORLD, "output.dat",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    
    // Write data (independent operation)
    MPI_File_write(fh, &buf, 1, MPI_INT, MPI_STATUS_IGNORE);
    
    // Close file (collective operation)
    MPI_File_close(&fh);
    
    MPI_Finalize();
    return 0;
}
```

**Expected Behavior**:
- Each process writes its rank to the file
- **PROBLEM**: All processes write to beginning of file (offset 0)
- Result: Only one rank's data appears (race condition)
- **This code is broken** - demonstrates why offsets are needed

### **API Details (Slide 19)**

**Key Points**:
- `MPI_File_open()` is **collective** on the communicator
- File mode is a **flag** (integer constant)
- `MPI_INFO_NULL`: No performance hints provided
- `MPI_File_write()` is **independent**
- `MPI_File_close()` is **collective** - if one process closes, all close

### **When to Use Independent I/O (Slide 20)**

**Appropriate Scenarios**:
- Reading/writing small portions of a file
- Overhead of collective calls is too high
- Processes have completely independent data
- No need for optimization across processes

**Example**: Reading a small configuration value that each process needs

### **Accessing Shared Files with Offsets (Slides 21-22)**

**Critical APIs**:

**1. MPI_File_seek()**:
```c
int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);
```
- Sets file pointer position (like C's `fseek`)
- **offset**: Position in bytes
- **whence**: Reference point
  - `MPI_SEEK_SET`: From beginning of file
  - `MPI_SEEK_CUR`: From current position
  - `MPI_SEEK_END`: From end of file

**2. MPI_File_write_at()**:
```c
int MPI_File_write_at(MPI_File fh, MPI_Offset offset,
                      const void *buf, int count,
                      MPI_Datatype datatype, MPI_Status *status);
```
- Combines seek + write in one operation
- **Thread-safe** (important for OpenMP)
- Specify offset directly in write call

**3. MPI_File_read_at()**:
```c
int MPI_File_read_at(MPI_File fh, MPI_Offset offset,
                     void *buf, int count,
                     MPI_Datatype datatype, MPI_Status *status);
```
- Combines seek + read in one operation
- Thread-safe

**Performance Note**: 
On modern parallel file systems (like Setonix), seek operations are well-optimized. On old HDDs, seeking could degrade performance due to physical head movement.

### **Example: Writing with Offsets (Slide 23)**

**Complete Working Code**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int buffer[8];  // 8 integers per process
    int avg_num_ints = 8;
    MPI_Offset offset;
    MPI_File fh;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize buffer with values based on rank
    for (int i = 0; i < avg_num_ints; i++) {
        buffer[i] = rank * avg_num_ints + i;
    }
    // Example: Rank 0: [0,1,2,3,4,5,6,7]
    //          Rank 1: [8,9,10,11,12,13,14,15]
    
    // Calculate offset for this process
    offset = rank * avg_num_ints * sizeof(int);
    
    // Open file
    MPI_File_open(MPI_COMM_WORLD, "datafile.dat",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    
    // Write at specific offset
    MPI_File_write_at(fh, offset, buffer, avg_num_ints,
                      MPI_INT, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}
```

**Key Calculation**:
```
offset = rank × elements_per_process × size_of_element
       = rank × 8 × 4 bytes
       = rank × 32 bytes

Rank 0: offset = 0
Rank 1: offset = 32
Rank 2: offset = 64
Rank 3: offset = 96
```

**Result**: Each process writes to different location - no race conditions!

### **Example: Reading Data Back (Slide 24)**

**Complete Reading Code**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int buffer[8];
    int avg_num_ints = 8;
    MPI_Offset offset;
    MPI_File fh;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate same offset as writing
    offset = rank * avg_num_ints * sizeof(int);
    
    // Open file for reading
    MPI_File_open(MPI_COMM_WORLD, "datafile.dat",
                  MPI_MODE_RDONLY,  // Read-only mode
                  MPI_INFO_NULL, &fh);
    
    // Seek to position
    MPI_File_seek(fh, offset, MPI_SEEK_SET);
    
    // Read data
    MPI_File_read(fh, buffer, avg_num_ints, MPI_INT,
                  MPI_STATUS_IGNORE);
    
    // Print what we read
    printf("Rank %d read: ", rank);
    for (int i = 0; i < avg_num_ints; i++) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
    
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}
```

**Output Example** (4 processes):
```
Rank 0 read: 0 1 2 3 4 5 6 7
Rank 1 read: 8 9 10 11 12 13 14 15
Rank 2 read: 16 17 18 19 20 21 22 23
Rank 3 read: 24 25 26 27 28 29 30 31
```

**Alternative Approach**:
Could use `MPI_File_read_at()` instead of seek + read:
```c
MPI_File_read_at(fh, offset, buffer, avg_num_ints,
                 MPI_INT, MPI_STATUS_IGNORE);
```

---

## **PART 5: Non-Contiguous I/O and File Views (Slides 25-35)**

### **Non-Contiguous Access Overview (Slide 25)**

**What is Non-Contiguous Access?**
Data that is not stored in consecutive memory locations or file positions.

**Example**:
```
Want to write: [A, B, _, _, _, C, D, _, _, _]
Where: A, B, C, D are data
       _ represents gaps (don't write)
```

**Why Common in Parallel Applications?**
- Distributed arrays stored in files
- Each process owns non-adjacent pieces
- Sub-arrays, slices, or strided access patterns

**Key Advantage of MPI**:
Can specify non-contiguous patterns efficiently without manually seeking multiple times.

### **File Views Concept (Slide 27)**

**File View Definition**:
A file view specifies which portion of the file is visible to a process.

**Specified by Three Components**:
```
1. Displacement: Offset from start (e.g., skip header)
2. Etype: Elementary data type (basic unit)
3. Filetype: Pattern of access (which parts visible)
```

**Created Using**:
```c
int MPI_File_set_view(MPI_File fh, MPI_Offset disp,
                      MPI_Datatype etype, MPI_Datatype filetype,
                      char *datarep, MPI_Info info);
```

**Components Explained**:

**1. Displacement**:
- Number of bytes to skip from beginning of file
- Typically used to skip file header
- Example: `displacement = 2020` skips 2020-byte header

**2. Etype (Elementary Type)**:
- Basic unit of data access
- Usually `MPI_INT`, `MPI_FLOAT`, etc.
- All accesses are in multiples of etype

**3. Filetype**:
- Specifies which portions are visible
- Can be simple (contiguous) or complex (non-contiguous)
- Created using MPI derived datatypes

### **Example: Non-Contiguous Write (Slide 28)**

**Scenario**:
```
Want to write 6 integers per process:
Process writes: [value1, value2, _, _, _, _]
                 ↑       ↑       ↑  gaps
                 2 integers      4 integer gaps

Then repeat pattern
```

**Visual Representation**:
```
File structure:
┌──────────────────────────────────────┐
│ Header (2020 bytes)                  │
├──────────────────────────────────────┤
│ Process 0: [val, val, _, _, _, _]   │
│ Process 1: [val, val, _, _, _, _]   │
│ Process 2: [val, val, _, _, _, _]   │
└──────────────────────────────────────┘
```

### **Complete Non-Contiguous Example (Slide 29-30)**

**Full Working Code**:
```c
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, buf[1000];
    MPI_File fh;
    MPI_Aint lb, extent;
    MPI_Datatype etype, filetype, contig;
    MPI_Offset disp;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Initialize buffer
    for (int i = 0; i < 1000; i++) {
        buf[i] = rank * 1000 + i;
    }
    
    // Step 1: Create contiguous type (2 integers)
    MPI_Type_contiguous(2, MPI_INT, &contig);
    
    // Step 2: Resize to create gaps
    // Lower bound = 0, extent = 6 integers
    lb = 0;
    extent = 6 * sizeof(int);  // 24 bytes
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);
    
    // Step 3: Set up displacement (skip header)
    disp = 2020;  // Skip 2020-byte header
    etype = MPI_INT;
    
    // Open file
    MPI_File_open(MPI_COMM_WORLD, "datafile",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    
    // Set file view
    MPI_File_set_view(fh, disp, etype, filetype,
                      "native", MPI_INFO_NULL);
    
    // Write data
    MPI_File_write(fh, buf, 1000, MPI_INT,
                   MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    MPI_Finalize();
    return 0;
}
```

**Step-by-Step Explanation**:

**Step 1: Create Contiguous Type**
```c
MPI_Type_contiguous(2, MPI_INT, &contig);
```
Creates type representing 2 consecutive integers.

**Step 2: Resize to Create Pattern**
```c
lb = 0;  // Lower bound at 0
extent = 6 * sizeof(int);  // Extent is 6 integers (24 bytes)
MPI_Type_create_resized(contig, lb, extent, &filetype);
```
- Takes the 2-integer type
- Sets its "extent" to 6 integers
- Result: Pattern repeats every 6 integers, but only first 2 contain data

**Visual**:
```
Original contig:  [INT][INT]
After resize:     [INT][INT][gap][gap][gap][gap]
                  ←---- extent (24 bytes) ----→
```

**Step 3: Set File View**
```c
MPI_File_set_view(fh, disp, etype, filetype, "native", MPI_INFO_NULL);
```
- `disp = 2020`: Skip 2020-byte header
- `etype = MPI_INT`: Basic unit is integer
- `filetype`: Non-contiguous pattern we created
- `"native"`: Data stored in native format (same as memory)

**Step 4: Write**
```c
MPI_File_write(fh, buf, 1000, MPI_INT, MPI_STATUS_IGNORE);
```
- Writes 1000 integers from buffer
- But file view determines WHERE they go
- Results in non-contiguous layout in file

**Verifying Output**:
```bash
# View binary file with specific format
xxd -s 2020 -c 24 -g 4 datafile

# -s 2020: Skip 2020 bytes (header)
# -c 24: Show 24 bytes per line (6 integers)
# -g 4: Group in 4-byte chunks (integers)
```

**Output Shows**:
```
Rank 0: 0000 0001 0000 0000 0000 0000
        ↑    ↑    ↑    gaps (zeros)
        data data

Rank 0: 0002 0003 0000 0000 0000 0000
        continuing pattern...
```

### **"native" Data Representation (Slide 30)**

**What "native" Means**:
- Data stored in same format as in memory
- No conversion or byte reordering
- Fastest option (no overhead)

**Other Options** (not covered in detail):
- `"internal"`: MPI-specific format
- `"external32"`: Portable across architectures
- Custom representations possible

**For This Course**: Always use `"native"` unless specifically told otherwise.

---

## **PART 6: Collective I/O Operations (Slides 36-43)**

### **Introduction to Collective I/O (Slide 36)**

**Key Benefit**: Provides **massive speedup** over independent I/O

**Characteristics**:
- All processes in communicator must participate
- Like collective communication operations
- Allows MPI library to optimize across processes
- Merges requests for efficiency

**Basic Idea**:
Build large blocks of data so reads/writes to I/O system are as large as possible - larger I/O operations are more efficient.

### **Why Collective I/O is Faster (Slide 37)**

**Optimization Technique**: Merging requests from different processes

**Especially Effective When**:
- Non-contiguous access patterns from each process
- Patterns overlap or interleave in file
- Multiple small requests can be combined into fewer large requests

**Visual Example**:
```
Independent I/O:
P0: ──read──  ──read──  ──read──
P1:   ──read──  ──read──  ──read──
     (many small I/O operations)

Collective I/O:
P0+P1: ────────read────────
       (one large coordinated operation)
```

### **Collective I/O APIs (Slide 38)**

**Main Function**:
```c
int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset,
                          const void *buf, int count,
                          MPI_Datatype datatype,
                          MPI_Status *status);
```

**Name Components**:
- `write`: Operation type
- `at`: Explicit offset specification
- `all`: Collective (all processes participate)

**Key Properties**:
- All processes must call
- Provides thread safety (important for OpenMP)
- Offset specified per process
- MPI coordinates across processes

### **Other Collective I/O Functions (Slide 39)**

**Full Set of Operations**:
```c
// With explicit offsets (thread-safe)
MPI_File_read_at_all()
MPI_File_write_at_all()

// Using file pointer
MPI_File_seek()
MPI_File_read_all()
MPI_File_write_all()

// Shared file pointer
MPI_File_read_shared()
MPI_File_write_shared()
MPI_File_read_ordered()
MPI_File_write_ordered()
```

**For Exams**: Focus on `_at_all` versions (most common)

---

## **PART 7: I/O Access Pattern Levels (Slides 40-43)**

### **Introduction to Access Levels (Slide 40)**

**Problem**: Same access pattern can be expressed in different ways

**Four Levels** (Level 0 → Level 3, increasing performance):
- **Level 0**: Unix-style (worst performance)
- **Level 1**: Independent I/O with communicators
- **Level 2**: Independent I/O with file views
- **Level 3**: Collective I/O with file views (best performance)

### **Example: 2D Array Distribution (Slide 41)**

**Scenario**:
Large 2D array distributed among 16 processes (4×4 grid)

```
┌────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │
├────┼────┼────┼────┤
│ P4 │ P5 │ P6 │ P7 │
├────┼────┼────┼────┤
│ P8 │ P9 │P10 │P11 │
├────┼────┼────┼────┤
│P12 │P13 │P14 │P15 │
└────┴────┴────┴────┘
```

**File Storage**: Row-major order
```
File: [P0_row1][P1_row1][P2_row1][P3_row1]
      [P0_row2][P1_row2][P2_row2][P3_row2]
      [P0_row3][P1_row3][P2_row3][P3_row3]
      [P4_row1][P4_row1]...
```

**Access Pattern**:
Each process needs to read its rows, which are non-contiguous in the file (interleaved with other processes' rows).

### **Level 0: Local Element Access (Slide 42)**

**Description**: Unix-style I/O
- Each process makes **one independent request per row**
- No MPI-specific optimizations
- Multiple small I/O operations

**Characteristics**:
- Only access local elements
- Remote access requires explicit MPI communication (not used for I/O)
- Worst performance

**Example Pattern**:
```
Process 0 reads:
  seek to row1, read
  seek to row2, read  
  seek to row3, read
  (3 separate I/O operations)
```

### **Level 1: Remote Element Access**

**Description**: Independent I/O with global communicator
- Same as Level 0 but allows global communicator
- Can access any element (even remote)
- Still uses MPI calls explicitly
- Multiple small operations

**Characteristics**:
- Global access via MPI communicator
- Still independent I/O (no coordination)
- Slightly better than Level 0

### **Level 2: Sub-array/Section Access**

**Description**: Independent I/O with derived datatypes and file views
- Create derived datatype for non-contiguous pattern
- Define file view
- Call independent I/O functions
- MPI can optimize within each process

**Characteristics**:
- Access rows/columns as sections
- File view describes pattern
- Better performance through datatype optimization
- Still independent (no cross-process coordination)

**Example**:
```c
// Create datatype for process's rows
MPI_Type_create_subarray(...);
MPI_File_set_view(fh, ..., subarray_type, ...);
MPI_File_read(...);  // Independent
```

### **Level 3: Full Global Access - BEST**

**Description**: Collective I/O with derived datatypes and file views
- Same setup as Level 2 (datatypes + views)
- Use **collective** I/O functions
- MPI coordinates across ALL processes
- **Highest performance**

**Characteristics**:
- Full global array access
- Collective operations merge requests
- Orders of magnitude faster
- Recommended approach

**Example**:
```c
// Create datatype for process's rows
MPI_Type_create_subarray(...);
MPI_File_set_view(fh, ..., subarray_type, ...);
MPI_File_read_all(...);  // COLLECTIVE - key difference!
```

### **Performance Comparison (Slide 43)**

**Graph Results** (from Blue Gene/Blue Waters supercomputers):

```
Performance (MB/s):
Level 0: ██ ~50 MB/s
Level 1: ███ ~100 MB/s  
Level 2: █████ ~200 MB/s
Level 3: ████████████████ ~800+ MB/s

Level 3 is 10-15× faster than Level 0!
```

**Key Takeaway**:
> "Collective input/output combined with non-contiguous access yields the highest performance"

**For Assignments/Exams**: Always prefer Level 3 (collective I/O) when possible.

---

## **PART 8: Process Topologies Introduction (Slides 47-52)**

### **Why Virtual Topologies? (Slide 48)**

**Problem with Default MPI**:
- Processes treated as linear range (0, 1, 2, 3, ...)
- Hardware is NOT a line of machines
- Numerical algorithms often have structured communication patterns

**Examples of Communication Structures**:
```
Tree:           2D Grid:        3D Grid:
    0              0─1─2          Multiple layers
   / \             │ │ │          of 2D grids
  1   2            3─4─5
 / \   \           │ │ │
3  4    5          6─7─8
```

**Solution**: Virtual topologies let you specify communication structure

**Benefits**:
- Simplify code (communicate with "neighbors" instead of calculating ranks)
- Improve performance (MPI can optimize mapping to hardware)
- Better match algorithm structure

### **Physical vs Virtual Topology (Slide 50)**

**Physical Topology**:
- Actual hardware connections
- How cores/chips/nodes are connected
- Fixed by the machine architecture

**Virtual Topology**:
- Logical communication pattern
- How your algorithm thinks about process connections
- Defined by programmer

**Mapping Problem**:
Need to map virtual topology onto physical topology efficiently. MPI handles this but respects hints you provide.

### **Two Types of Virtual Topologies (Slide 51)**

**1. Cartesian Topologies** (most common):
- Multi-dimensional grids
- Regular structure
- Use `MPI_Cart_*` functions
- Examples: 2D grid, 3D grid, rings, torus

**2. Graph Topologies** (more general):
- Arbitrary connections
- Irregular structure  
- Use `MPI_Graph_*` or `MPI_Dist_graph_*` functions
- Examples: Trees, arbitrary networks

---

## **PART 9: Cartesian Topologies (Slides 52-65)**

### **Cartesian Topology Basics (Slide 52)**

**What is Cartesian Topology?**
Processes arranged in multi-dimensional grid with optional periodic boundaries.

**Default: MPI_COMM_WORLD**
```
Rank: 0 ─ 1 ─ 2 ─ 3 ─ 4 ─ 5
(1D linear arrangement)
```

**Cartesian Examples**:

**1D Ring (periodic)**:
```
0 ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 0 (wraps around)
└─────────────────────┘
```

**2D Grid (non-periodic)**:
```
0 ─ 1 ─ 2
│   │   │
3 ─ 4 ─ 5
│   │   │
6 ─ 7 ─ 8
```

**2D Cylinder (periodic in one dimension)**:
```
0 ─ 1 ─ 2     Wraps horizontally:
│   │   │     0←→2, 3←→5, 6←→8
3 ─ 4 ─ 5     But NOT vertically
│   │   │
6 ─ 7 ─ 8
```

**2D Torus (periodic in both dimensions)**:
```
0 ─ 1 ─ 2     Wraps both ways:
│   │   │     Horizontal: 0←→2
3 ─ 4 ─ 5     Vertical: 0←→6
│   │   │     
6 ─ 7 ─ 8     (Topologically a donut)
└───┴───┘
```

### **MPI_Cart_create API (Slide 55)**

**Function Signature**:
```c
int MPI_Cart_create(MPI_Comm comm_old, int ndims,
                    const int dims[], const int periods[],
                    int reorder, MPI_Comm *comm_cart);
```

**Parameters**:

**comm_old (IN)**: 
- Input communicator (usually MPI_COMM_WORLD)

**ndims (IN)**: 
- Number of dimensions (1D, 2D, 3D, etc.)

**dims (IN)**: 
- Array of integers specifying grid size in each dimension
- Example: `dims[] = {4, 3}` creates 4×3 grid (12 processes)

**periods (IN)**: 
- Array specifying if each dimension is periodic
- `0` or `false`: Non-periodic (edges don't connect)
- `1` or `true`: Periodic (wraps around)
- Example: `periods[] = {1, 0}` = periodic in dim 0, not in dim 1

**reorder (IN)**: 
- Whether MPI can reorder ranks for optimization
- Usually set to `1` (true) to allow optimization

**comm_cart (OUT)**: 
- New communicator with Cartesian topology

**Important Notes**:
- All processes must call with same arguments
- If total grid size < number of processes, some get MPI_COMM_NULL
- If grid size > number of processes, returns error

### **Cartesian Examples (Slides 56-59)**

**Example 1: 1D Ring (Slide 56)**
```c
int ndims = 1;
int dims[1] = {4};      // 4 processes in 1D
int periods[1] = {1};   // Periodic (ring)
int reorder = 1;        // Allow reordering
MPI_Comm comm_cart;

MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                reorder, &comm_cart);

// Result: 0 ─ 1 ─ 2 ─ 3 ─ 0 (circular)
```

**Example 2: 2D Grid (Non-periodic) (Slide 57)**
```c
int ndims = 2;
int dims[2] = {3, 3};       // 3×3 grid
int periods[2] = {0, 0};    // Non-periodic both directions
int reorder = 1;
MPI_Comm comm_cart;

MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                reorder, &comm_cart);

// Result: 
// 0 ─ 1 ─ 2
// │   │   │
// 3 ─ 4 ─ 5
// │   │   │
// 6 ─ 7 ─ 8
```

**Example 3: 2D Cylinder (Periodic in vertical only) (Slide 58)**
```c
int ndims = 2;
int dims[2] = {3, 3};
int periods[2] = {1, 0};    // Periodic in dim 0, not dim 1
int reorder = 1;
MPI_Comm comm_cart;

MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                reorder, &comm_cart);

// Wraps vertically: 0↔6, 1↔7, 2↔8
// Does NOT wrap horizontally
```

**Example 4: 2D Torus (Periodic both directions) (Slide 58)**
```c
int ndims = 2;
int dims[2] = {3, 3};
int periods[2] = {1, 1};    // Periodic both dimensions
int reorder = 1;
MPI_Comm comm_cart;

MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                reorder, &comm_cart);

// Wraps both horizontally and vertically
// Forms a torus (donut shape topologically)
```

**Example 5: 3D Torus (Slide 59)**
```c
int ndims = 3;
int dims[3] = {2, 2, 2};     // 2×2×2 cube
int periods[3] = {1, 1, 1};  // Periodic all 3 dimensions
int reorder = 1;
MPI_Comm comm_cart;

MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                reorder, &comm_cart);

// 3D torus (harder to visualize!)
// 8 processes in 3D with all dimensions periodic
```

### **Working with Cartesian Coordinates (Slide 60)**

**Complete 2D Torus Example**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int ndims = 2;
    int dims[2] = {4, 3};        // 4×3 grid = 12 processes
    int periods[2] = {1, 1};     // Torus (periodic both ways)
    int reorder = 1;
    int coords[2];
    MPI_Comm comm_cart;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                    reorder, &comm_cart);
    
    // Get coordinates for this rank
    MPI_Cart_coords(comm_cart, rank, ndims, coords);
    
    printf("Rank %d has coordinates (%d, %d)\n",
           rank, coords[0], coords[1]);
    
    MPI_Finalize();
    return 0;
}
```

**Output**:
```
Rank 0 has coordinates (0, 0)
Rank 1 has coordinates (0, 1)
Rank 2 has coordinates (0, 2)
Rank 3 has coordinates (1, 0)
Rank 4 has coordinates (1, 1)
...
Rank 11 has coordinates (3, 2)
```

### **Coordinate Inquiry Functions (Slide 61)**

**MPI_Cart_coords** - Get coordinates from rank:
```c
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims,
                    int coords[]);
```
- **Input**: rank
- **Output**: Cartesian coordinates for that rank

**MPI_Cart_rank** - Get rank from coordinates:
```c
int MPI_Cart_rank(MPI_Comm comm, const int coords[],
                  int *rank);
```
- **Input**: Cartesian coordinates
- **Output**: rank corresponding to those coordinates

**Example Usage**:
```c
int rank, coords[2];
MPI_Comm_rank(comm_cart, &rank);

// Get my coordinates
MPI_Cart_coords(comm_cart, rank, 2, coords);
printf("I am rank %d at (%d,%d)\n", rank, coords[0], coords[1]);

// Find rank of neighbor
int neighbor_coords[2] = {coords[0], coords[1] + 1};
int neighbor_rank;
MPI_Cart_rank(comm_cart, neighbor_coords, &neighbor_rank);
printf("My right neighbor is rank %d\n", neighbor_rank);
```

### **MPI_Dims_create - Automatic Grid Sizing (Slide 63)**

**Purpose**: Automatically determine balanced grid dimensions

**Function Signature**:
```c
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
```

**How It Works**:
- **nnodes**: Number of processes to distribute
- **ndims**: Number of dimensions
- **dims**: Array with dimension constraints (0 = auto, >0 = fixed)

**Examples**:

**Example 1**: 6 processes, 2D, no constraints
```c
int nnodes = 6;
int ndims = 2;
int dims[2] = {0, 0};  // Both dimensions auto

MPI_Dims_create(nnodes, ndims, dims);
// Result: dims = {3, 2}  (3×2 grid)
```

**Example 2**: 7 processes, 2D (impossible to factor evenly)
```c
int nnodes = 7;
int ndims = 2;
int dims[2] = {0, 0};

MPI_Dims_create(nnodes, ndims, dims);
// Result: dims = {7, 1}  (best it can do)
```

**Example 3**: 6 processes, 3D, middle dimension fixed
```c
int nnodes = 6;
int ndims = 3;
int dims[3] = {0, 3, 0};  // Middle dimension must be 3

MPI_Dims_create(nnodes, ndims, dims);
// Result: dims = {2, 3, 1}  (2×3×1)
```

**Example 4**: 7 processes, 3D, constraint can't be satisfied
```c
int nnodes = 7;
int ndims = 3;
int dims[3] = {0, 3, 0};  // Want middle = 3, but 7/3 doesn't work

MPI_Dims_create(nnodes, ndims, dims);
// Result: ERROR - returns error code
```

### **MPI_Cart_shift - Finding Neighbors (Slide 64)**

**Purpose**: Get ranks of neighboring processes in a direction

**Function Signature**:
```c
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp,
                   int *rank_source, int *rank_dest);
```

**Parameters**:
- **comm**: Cartesian communicator
- **direction**: Which dimension (0=first, 1=second, etc.)
- **disp**: Displacement
  - `disp > 0`: Upward shift
  - `disp < 0`: Downward shift
  - `disp = 1`: Immediate neighbor
- **rank_source (OUT)**: Rank of source neighbor
- **rank_dest (OUT)**: Rank of destination neighbor

**Example**:
```c
int rank, up, down, left, right;
MPI_Comm_rank(comm_cart, &rank);

// Get vertical neighbors (dimension 0)
MPI_Cart_shift(comm_cart, 0, 1, &down, &up);

// Get horizontal neighbors (dimension 1)
MPI_Cart_shift(comm_cart, 1, 1, &left, &right);

printf("Rank %d: up=%d, down=%d, left=%d, right=%d\n",
       rank, up, down, left, right);
```

**Visual for 3×3 Grid**:
```
Rank 4 (center):
    up=1
     ↑
left=3 ← [4] → right=5
     ↓
   down=7
```

**Boundary Handling**:
- **Non-periodic**: Neighbors outside grid return `MPI_PROC_NULL`
- **Periodic**: Wraps around to opposite side

### **Simple Cart_shift Example (Slide 65)**

**Complete Code**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank;
    int ndims = 1;
    int dims[1] = {4};      // 1D with 4 processes
    int periods[1] = {1};   // Periodic (ring)
    int reorder = 1;
    MPI_Comm comm_cart;
    int left, right;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create 1D ring
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                    reorder, &comm_cart);
    
    // Get neighbors
    MPI_Cart_shift(comm_cart, 0, 1, &left, &right);
    
    printf("Rank %d: left=%d, right=%d\n", rank, left, right);
    
    MPI_Finalize();
    return 0;
}
```

**Output** (4 processes in ring):
```
Rank 0: left=3, right=1  (wraps to 3 on left)
Rank 1: left=0, right=2
Rank 2: left=1, right=3
Rank 3: left=2, right=0  (wraps to 0 on right)
```

---

## **PART 10: Graph Topologies (Slides 66-76)**

### **Graph Topology Introduction (Slide 66)**

**General Graph Topology**:
Most flexible way to define arbitrary communication patterns.

**Graph Components**:
- **Nodes**: Processes
- **Edges**: Communication links between processes
- **Directed edges**: Show origin and destination

**Visual Example**:
```
    0
   ↙ ↓
  1 → 2
      ↓
      3
```

**How to Specify**:
Use two arrays:
1. **Index array**: Cumulative count of edges
2. **Edges array**: List of connected neighbors

### **Graph Specification Details (Slide 67)**

**Index Array**:
Describes node degree (number of connections) cumulatively.

**Example Graph**:
```
Node 0: connects to [1]      (1 edge)
Node 1: connects to [0, 2]   (2 edges)
Node 2: connects to [1, 3]   (2 edges)
Node 3: connects to [2]      (1 edge)
```

**Index Array Construction**:
```
index[0] = 1   (node 0 has 1 edge, cumulative = 1)
index[1] = 3   (node 1 has 2 edges, cumulative = 1+2 = 3)
index[2] = 5   (node 2 has 2 edges, cumulative = 3+2 = 5)
index[3] = 6   (node 3 has 1 edge, cumulative = 5+1 = 6)
```

**Edges Array**:
```
edges[] = {1, 0, 2, 1, 3, 2}
           ↑  ↑  ↑  ↑  ↑  ↑
           │  └──┴──┘  └──┴─── Node 2 connects to
           │     Node 1 connects to
           └─ Node 0 connects to
```

**Reading the Arrays**:
- Node 0: Use edges[0:1] = {1}
- Node 1: Use edges[1:3] = {0, 2}
- Node 2: Use edges[3:5] = {1, 3}
- Node 3: Use edges[5:6] = {2}

### **MPI_Graph_create API (Slide 68)**

**Function Signature**:
```c
int MPI_Graph_create(MPI_Comm comm_old, int nnodes,
                     const int index[], const int edges[],
                     int reorder, MPI_Comm *comm_graph);
```

**Parameters**:
- **comm_old (IN)**: Input communicator
- **nnodes (IN)**: Number of nodes in graph
- **index (IN)**: Array of cumulative edge counts
- **edges (IN)**: Array of neighbor ranks
- **reorder (IN)**: Allow rank reordering (usually 1)
- **comm_graph (OUT)**: New graph communicator

**Note**: Usually better to use `MPI_Dist_graph_create` (distributed version) for large graphs, but concepts are the same.

### **Example: Creating Simple Graph (Slide 69)**

**Graph Structure**:
```
0 → 1 → 2 → 3
    ↑       ↓
    └───────┘
```

**Code**:
```c
int nnodes = 4;
int index[4] = {1, 3, 5, 6};
int edges[6] = {1, 0, 2, 1, 3, 2};
int reorder = 1;
MPI_Comm comm_graph;

MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges,
                 reorder, &comm_graph);
```

**Interpretation**:
- Node 0 → 1
- Node 1 → 0, 2
- Node 2 → 1, 3
- Node 3 → 2

### **Graph Neighbor Inquiry (Slide 70)**

**MPI_Graph_neighbors_count**:
```c
int MPI_Graph_neighbors_count(MPI_Comm comm, int rank,
                              int *nneighbors);
```
Returns number of neighbors for given rank.

**MPI_Graph_neighbors**:
```c
int MPI_Graph_neighbors(MPI_Comm comm, int rank,
                        int maxneighbors, int neighbors[]);
```
Returns array of neighbor ranks.

**Example Usage**:
```c
int rank, nneighbors;
MPI_Comm_rank(comm_graph, &rank);

// How many neighbors?
MPI_Graph_neighbors_count(comm_graph, rank, &nneighbors);

// Get their ranks
int *neighbors = malloc(nneighbors * sizeof(int));
MPI_Graph_neighbors(comm_graph, rank, nneighbors, neighbors);

printf("Rank %d has %d neighbors:", rank, nneighbors);
for (int i = 0; i < nneighbors; i++) {
    printf(" %d", neighbors[i]);
}
printf("\n");
```

### **Neighbor Collectives (Slide 71-72)**

**What Are Neighbor Collectives?**
Collective operations that only involve a process and its neighbors (not all processes).

**Example: MPI_Neighbor_allgather**:
```c
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount,
                          MPI_Datatype sendtype,
                          void *recvbuf, int recvcount,
                          MPI_Datatype recvtype,
                          MPI_Comm comm);
```

**Behavior**:
- Each process sends data to all its neighbors
- Each process receives data from all its neighbors
- Only neighbors participate (not entire communicator)

**Use Case Example**:
```
In stencil computations:
    ┌───┬───┬───┐
    │   │ N │   │
    ├───┼───┼───┤
    │ W │ P │ E │  Process P exchanges with neighbors
    ├───┼───┼───┤
    │   │ S │   │
    └───┴───┴───┘

MPI_Neighbor_allgather sends P's data to N,S,E,W
                       and receives their data
```

### **Complete Neighbor Collective Example (Slide 73)**

**Graph Structure**:
```
0 ↔ 1 ↔ 2
    ↕
    3
```

**Complete Code**:
```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    int nnodes = 4;
    int index[4] = {1, 3, 4, 5};
    int edges[5] = {1, 0, 2, 1, 1};  // Custom graph
    int reorder = 1;
    MPI_Comm comm_graph;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create graph topology
    MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges,
                     reorder, &comm_graph);
    
    // Get number of neighbors
    int nneighbors;
    MPI_Graph_neighbors_count(comm_graph, rank, &nneighbors);
    
    // Prepare send/receive buffers
    int sendbuf = rank * 100;  // My data
    int *recvbuf = malloc(nneighbors * sizeof(int));
    
    // Neighbor all-gather
    MPI_Neighbor_allgather(&sendbuf, 1, MPI_INT,
                          recvbuf, 1, MPI_INT,
                          comm_graph);
    
    // Print received data
    printf("Rank %d received from neighbors:", rank);
    for (int i = 0; i < nneighbors; i++) {
        printf(" %d", recvbuf[i]);
    }
    printf("\n");
    
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
```

**Output**:
```
Rank 0 received from neighbors: 100        (from rank 1)
Rank 1 received from neighbors: 0 200 300  (from ranks 0,2,3)
Rank 2 received from neighbors: 100        (from rank 1)
Rank 3 received from neighbors: 100        (from rank 1)
```

---

## **PART 11: Summary and Key Takeaways (Slide 78)**

### **Summary Points**

**1. I/O Performance is Critical**:
- I/O can destroy performance if handled poorly
- Always consider I/O optimization in HPC applications
- Use parallel I/O whenever possible

**2. MPI I/O Offers Multiple Approaches**:
- **Independent file access**: Simple, limited performance
- **Non-contiguous access**: For complex data patterns
- **Collective calls**: Best performance (orders of magnitude faster)

**3. File Views Enable Efficiency**:
- Describe non-contiguous patterns with datatypes
- Set once, use many times
- MPI optimizes based on view

**4. Virtual Topologies Improve Code**:
- Simplify communication patterns
- Allow MPI to optimize mapping
- Match algorithm structure to communication

**5. Cartesian Grids Are Common**:
- Flexible and widely applicable
- Support periodic boundaries (rings, torus)
- Useful for numerical routines (PDEs, simulations)

**6. Graph Topologies Are General**:
- Can represent any communication pattern
- More complex to set up
- Neighbor collectives are powerful

---

## **Exam Preparation Tips**

### **Must-Know Concepts**

**1. I/O Strategies**:
- Difference between non-parallel, independent, and cooperative I/O
- Why collective I/O is faster
- When to use each approach

**2. Key MPI I/O Functions**:
```c
MPI_File_open()      // Collective
MPI_File_close()     // Collective
MPI_File_write()     // Independent
MPI_File_write_at()  // Independent with offset
MPI_File_write_all() // Collective
MPI_File_read_at_all() // Collective with offset
MPI_File_seek()      // Set file position
MPI_File_set_view()  // Define file view
```

**3. File Views**:
- **Displacement**: Skip header
- **Etype**: Basic unit (MPI_INT, etc.)
- **Filetype**: Access pattern (contiguous or derived type)

**4. Four Levels of Access**:
- Level 0: Local element (worst)
- Level 1: Remote element with communicators
- Level 2: Sub-array with file views
- Level 3: Collective with file views (best - 10-15× faster)

**5. Cartesian Topology Functions**:
```c
MPI_Cart_create()      // Create Cartesian communicator
MPI_Cart_coords()      // Rank → coordinates
MPI_Cart_rank()        // Coordinates → rank
MPI_Cart_shift()       // Find neighbors
MPI_Dims_create()      // Auto-size grid
```

**6. Graph Topology Functions**:
```c
MPI_Graph_create()            // Create graph
MPI_Graph_neighbors_count()   // Count neighbors
MPI_Graph_neighbors()         // Get neighbor ranks
MPI_Neighbor_allgather()      // Collective among neighbors
```

### **Common Exam Questions**

**Type 1: Code Analysis**
"What does this MPI I/O code do?"
- Trace through offsets
- Identify independent vs collective
- Determine what gets written where

**Type 2: Fix Broken Code**
"This code has a race condition. Fix it."
- Add offsets with MPI_File_seek or MPI_File_write_at
- Change independent to collective operations
- Fix topology parameters

**Type 3: Design Questions**
"Design a parallel I/O strategy for distributing a large matrix."
- Choose collective I/O (Level 3)
- Create derived datatypes for sub-arrays
- Set appropriate file views
- Use MPI_File_write_at_all

**Type 4: Topology Questions**
"Create a 3D torus topology for 8 processes."
- ndims = 3
- dims = {2, 2, 2}
- periods = {1, 1, 1}
- Show rank to coordinate mapping

**Type 5: Performance Questions**
"Why is collective I/O faster than independent I/O?"
- Merges multiple small requests
- Reduces number of I/O operations
- Better utilization of parallel file system
- Orders of magnitude improvement

### **Common Mistakes to Avoid**

**1. Forgetting Offsets**:
```c
// WRONG - Race condition
MPI_File_write(fh, &buf, 1, MPI_INT, MPI_STATUS_IGNORE);

// CORRECT - Use offset
offset = rank * sizeof(int);
MPI_File_write_at(fh, offset, &buf, 1, MPI_INT, MPI_STATUS_IGNORE);
```

**2. Mixing Collective/Independent**:
```c
// WRONG - Not all processes call collective
if (rank == 0) {
    MPI_File_write_all(...);  // Only rank 0 calls
}

// CORRECT - All processes must call
MPI_File_write_all(...);  // All ranks call
```

**3. Wrong File Mode**:
```c
// WRONG - Can't write in read-only mode
MPI_File_open(..., MPI_MODE_RDONLY, ...);
MPI_File_write(...);  // ERROR!

// CORRECT - Use write mode
MPI_File_open(..., MPI_MODE_WRONLY | MPI_MODE_CREATE, ...);
MPI_File_write(...);
```

**4. Forgetting File Exists**:
```c
// WRONG - File doesn't exist, no CREATE flag
MPI_File_open(..., MPI_MODE_WRONLY, ...);  // ERROR!

// CORRECT - Add CREATE flag
MPI_File_open(..., MPI_MODE_WRONLY | MPI_MODE_CREATE, ...);
```

**5. Periodic Boundary Confusion**:
```c
// WRONG - 0 means FALSE (non-periodic)
int periods[2] = {0, 0};  // Not periodic!

// For periodic (torus):
int periods[2] = {1, 1};  // TRUE = periodic
```

**6. Topology Size Mismatch**:
```c
// WRONG - 3×3 = 9 processes, but only 4 running
int dims[2] = {3, 3};  // Need 9 processes
// With 4 processes → ERROR

// CORRECT - Match process count
int dims[2] = {2, 2};  // Need 4 processes
```

---

## **Practice Problems**

### **Problem 1: Basic I/O with Offsets**

**Question**: Write code where each of 4 processes writes its rank to a file at the correct offset.

**Solution**:
```c
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, buf;
    MPI_File fh;
    MPI_Offset offset;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    buf = rank;
    offset = rank * sizeof(int);
    
    MPI_File_open(MPI_COMM_WORLD, "output.dat",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    
    MPI_File_write_at(fh, offset, &buf, 1, MPI_INT,
                      MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}
```

### **Problem 2: Non-Contiguous Write**

**Question**: Each process writes 3 integers with 2-integer gaps between them.

**Solution**:
```c
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, buf[100];
    MPI_File fh;
    MPI_Datatype filetype, contig;
    MPI_Aint lb, extent;
    MPI_Offset disp = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Initialize buffer
    for (int i = 0; i < 100; i++) buf[i] = rank * 100 + i;
    
    // Create pattern: 3 ints, then 2-int gap (total extent = 5 ints)
    MPI_Type_contiguous(3, MPI_INT, &contig);
    lb = 0;
    extent = 5 * sizeof(int);
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);
    
    MPI_File_open(MPI_COMM_WORLD, "datafile",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    
    MPI_File_set_view(fh, disp, MPI_INT, filetype,
                      "native", MPI_INFO_NULL);
    
    MPI_File_write(fh, buf, 100, MPI_INT, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    MPI_Finalize();
    return 0;
}
```

### **Problem 3: 2D Grid Topology**

**Question**: Create a 3×4 non-periodic 2D grid and print each process's coordinates.

**Solution**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank;
    int ndims = 2;
    int dims[2] = {3, 4};      // 3×4 grid
    int periods[2] = {0, 0};   // Non-periodic
    int reorder = 1;
    int coords[2];
    MPI_Comm comm_cart;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                    reorder, &comm_cart);
    
    if (comm_cart != MPI_COMM_NULL) {
        MPI_Cart_coords(comm_cart, rank, ndims, coords);
        printf("Rank %d is at position (%d, %d)\n",
               rank, coords[0], coords[1]);
    }
    
    MPI_Finalize();
    return 0;
}
```

### **Problem 4: Finding Neighbors**

**Question**: In a 2D torus (4×4), find all neighbors of process at (1,1).

**Solution**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank;
    int ndims = 2;
    int dims[2] = {4, 4};      // 4×4 grid
    int periods[2] = {1, 1};   // Torus
    int reorder = 1;
    int coords[2];
    int up, down, left, right;
    MPI_Comm comm_cart;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods,
                    reorder, &comm_cart);
    
    MPI_Cart_coords(comm_cart, rank, ndims, coords);
    
    // Get vertical neighbors
    MPI_Cart_shift(comm_cart, 0, 1, &down, &up);
    
    // Get horizontal neighbors
    MPI_Cart_shift(comm_cart, 1, 1, &left, &right);
    
    if (coords[0] == 1 && coords[1] == 1) {
        printf("Process at (1,1) [rank %d] has neighbors:\n", rank);
        printf("  Up: %d, Down: %d, Left: %d, Right: %d\n",
               up, down, left, right);
    }
    
    MPI_Finalize();
    return 0;
}
```

**Expected**: Process at (1,1) connects to (0,1), (2,1), (1,0), (1,2)

### **Problem 5: Simple Graph Topology**

**Question**: Create a line graph: 0→1→2→3

**Solution**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank;
    int nnodes = 4;
    int index[4] = {1, 2, 3, 4};   // Cumulative edges
    int edges[4] = {1, 2, 3, 0};   // Connections
    int reorder = 1;
    int nneighbors;
    MPI_Comm comm_graph;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges,
                     reorder, &comm_graph);
    
    MPI_Graph_neighbors_count(comm_graph, rank, &nneighbors);
    
    int *neighbors = malloc(nneighbors * sizeof(int));
    MPI_Graph_neighbors(comm_graph, rank, nneighbors, neighbors);
    
    printf("Rank %d connects to:", rank);
    for (int i = 0; i < nneighbors; i++) {
        printf(" %d", neighbors[i]);
    }
    printf("\n");
    
    free(neighbors);
    MPI_Finalize();
    return 0;
}
```

---

## **Assignment Tips for Lecture 10 Concepts**

### **When to Use These Techniques**

**Use Parallel I/O when**:
- Writing checkpoint files
- Saving simulation results
- Processing large datasets
- Multiple processes produce output

**Use Collective I/O when**:
- All processes write/read at once
- Performance is critical
- File system supports parallel I/O (Setonix does!)

**Use Topologies when**:
- Implementing stencil computations (heat equation, wave equation)
- Matrix operations requiring neighbor communication
- Structured domain decomposition
- Grid-based simulations

### **Performance Considerations**

**For Setonix**:
- Has optimized parallel file system (Lustre)
- Collective I/O highly recommended
- No need to worry about old HDD optimizations
- Use MPI-IO over standard C I/O

**For Kaya**:
- Limited to 2-4 processes reliably
- Still benefits from parallel I/O
- Test thoroughly before submission

**General Tips**:
- Always use offsets to avoid race conditions
- Prefer collective over independent I/O
- Create derived datatypes for complex patterns
- Use MPI_File_write_at_all when possible

---

## **Quick Reference: Function Summary**

### **File Operations**
```c
// Opening/Closing (COLLECTIVE)
MPI_File_open(comm, filename, mode, info, &fh);
MPI_File_close(&fh);

// Modes (combine with |)
MPI_MODE_RDONLY     // Read only
MPI_MODE_WRONLY     // Write only
MPI_MODE_RDWR       // Read and write
MPI_MODE_CREATE     // Create if doesn't exist

// Independent I/O
MPI_File_write(fh, buf, count, datatype, status);
MPI_File_read(fh, buf, count, datatype, status);
MPI_File_seek(fh, offset, whence);
MPI_File_write_at(fh, offset, buf, count, datatype, status);
MPI_File_read_at(fh, offset, buf, count, datatype, status);

// Collective I/O
MPI_File_write_all(fh, buf, count, datatype, status);
MPI_File_read_all(fh, buf, count, datatype, status);
MPI_File_write_at_all(fh, offset, buf, count, datatype, status);
MPI_File_read_at_all(fh, offset, buf, count, datatype, status);

// File Views
MPI_File_set_view(fh, disp, etype, filetype, datarep, info);
```

### **Cartesian Topology**
```c
// Create
MPI_Cart_create(comm_old, ndims, dims[], periods[], 
                reorder, &comm_cart);

// Query
MPI_Cart_coords(comm, rank, maxdims, coords[]);
MPI_Cart_rank(comm, coords[], &rank);
MPI_Cart_shift(comm, direction, disp, &source, &dest);

// Utilities
MPI_Dims_create(nnodes, ndims, dims[]);
```

### **Graph Topology**
```c
// Create
MPI_Graph_create(comm_old, nnodes, index[], edges[],
                 reorder, &comm_graph);

// Query
MPI_Graph_neighbors_count(comm, rank, &nneighbors);
MPI_Graph_neighbors(comm, rank, maxneighbors, neighbors[]);

// Collectives
MPI_Neighbor_allgather(sendbuf, sendcount, sendtype,
                       recvbuf, recvcount, recvtype, comm);
```

---

## **Additional Resources**

**Official Documentation**:
- MPI Standard: https://www.mpi-forum.org/docs/
- MPI Tutorial: https://mpitutorial.com/
- Introduction to Parallel I/O: See lecture references

**Setonix Specific**:
- Parallel file system optimized for MPI-IO
- Use collective I/O for best performance
- Access via Pawsey account (check with instructor)

**Testing on Kaya**:
- Limited to 2 processes reliably
- Good for testing logic
- May not show full parallel performance

**Tips for Success**:
1. Start simple (independent I/O) before moving to collective
2. Test with small data sizes first
3. Always check return codes and status
4. Use appropriate file modes (CREATE, WRONLY, etc.)
5. Don't forget offsets!
6. Verify output files to ensure correctness

---

## **Final Checklist for Exam**

### **Must Know**

- [ ] Three I/O strategies (non-parallel, independent, cooperative)
- [ ] Difference between independent and collective I/O
- [ ] How to calculate offsets (rank × elements × size)
- [ ] File modes (RDONLY, WRONLY, RDWR, CREATE)
- [ ] MPI_File_open is collective
- [ ] MPI_File_close is collective
- [ ] Four levels of I/O access patterns
- [ ] Level 3 (collective + file views) is fastest
- [ ] What file views are (displacement, etype, filetype)
- [ ] How to create Cartesian topology
- [ ] What periodic boundaries mean (0=false, 1=true)
- [ ] How to find neighbors with MPI_Cart_shift
- [ ] How to create graph topology (index + edges arrays)
- [ ] Purpose of virtual topologies

### **Should Understand**

- [ ] Why collective I/O is faster (merges requests)
- [ ] How non-contiguous I/O works
- [ ] MPI_Type_create_resized for gaps
- [ ] Difference between Cartesian and graph topologies
- [ ] When to use each topology type
- [ ] Neighbor collectives concept
- [ ] Performance implications of I/O choices

### **Common Exam Traps**

- [ ] Forgetting offsets causes race conditions
- [ ] Mixing collective/independent calls
- [ ] Wrong file mode for operation
- [ ] Confusing periodic=0 (FALSE) vs periodic=1 (TRUE)
- [ ] Grid size doesn't match process count
- [ ] Not all processes calling collective operations
- [ ] Incorrect index/edges arrays for graphs

---

## **Last Minute Review - Key Formulas**

**Offset Calculation**:
```
offset = rank × elements_per_process × sizeof(datatype)
```

**Extent for Gaps**:
```
extent = (data_elements + gap_elements) × sizeof(datatype)
```

**Graph Index Array**:
```
index[i] = cumulative count of edges up to node i
```

**Cartesian Rank ↔ Coords**:
```
Use MPI_Cart_coords(comm, rank, ndims, coords)
Use MPI_Cart_rank(comm, coords, &rank)
```

---

## **Good Luck!**

Remember:
- **Collective I/O** is almost always better than independent
- **Always use offsets** to avoid race conditions
- **File views** enable efficient non-contiguous access
- **Topologies** simplify neighbor communication
- **Level 3 access** (collective + file views) gives best performance
- **Test your code** with small examples first

This material is examinable and commonly appears in HPC assignments and exams. Practice the code examples and understand the concepts rather than memorizing syntax.