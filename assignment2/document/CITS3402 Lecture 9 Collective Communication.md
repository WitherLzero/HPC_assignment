# CITS3402 Lecture 9: Collective Communication

## Study Notes for High Performance Computing

---

## **Key Insights Box**
**Main Learning Outcomes:**
- Master all MPI collective communication patterns (broadcast, scatter, gather, allgather, alltoall)
- Understand the performance advantages of collective operations over point-to-point calls
- Learn MPI groups and communicators for organizing process hierarchies
- Apply global reduction operations (sum, max, min, custom) across processes
- Design and implement custom communicators for complex multi-group applications
- Recognize when collective communication provides significant performance benefits

---

## **PART 1: Introduction to Collective Communication (Slides 1-8)**

### **What is Collective Communication? (Slide 6)**

**Definition**: Collective communications are MPI operations that transmit data among **all processes** in a communicator simultaneously, not just between process pairs.

**Core Principles**:
```
Collective Communication Characteristics:
├── ALL processes in communicator MUST participate
├── Provides implicit synchronization (barrier-like behavior)
├── Offers various data movement patterns (broadcast, scatter, gather)
├── Performs global reduction operations (sum, max, min, etc.)
├── Leverages special optimizations unavailable to send/recv
└── More efficient than implementing same pattern with point-to-point
```

**Why Collective Communication Matters**:
- **Performance**: Communication functions and communicator work together to achieve tremendous performance
- **Optimization**: Can leverage special optimizations over many point-to-point calls
- **Simplicity**: Replaces complex send/recv topologies with single function calls
- **Correctness**: Reduces chances of deadlock and race conditions

### **Semantics of Collective Operations (Slide 7)**

**Key Semantic Rules**:

**1. Root Process Concept**:
```
Some collective communication involves single process (root) coordinating:
├── Root typically has rank == 0
├── Root sends information to all others (broadcast, scatter)
├── Root receives information from all others (gather, reduce)
└── Some operations have no root (allgather, allreduce, barrier)
```

**2. Two Flavors of Collective Functions**:
```
Simple (Contiguous Data):
├── Data stored contiguously in memory
├── Simpler to use
├── Most common in practice
└── Example: float array[1000]

Vectored (Non-contiguous Data):
├── Can 'pick and choose' from array
├── More complex but more flexible
├── Useful for custom data layouts
└── Example: Every 3rd element, or struct members
```

### **Three Types of Collective Communication (Slide 8)**

**Type 1: Barrier Synchronization**
- Blocks until all processes have reached a synchronization point
- No data movement, pure synchronization
- Like a checkpoint all processes must reach

**Type 2: Data Movement (Global Communication)**
- Broadcast: One → All
- Scatter: One → All (different data to each)
- Gather: All → One
- Allgather: All → All
- Alltoall: All ↔ All

**Type 3: Collective Operations (Global Reductions)**
- One process collects data from each process
- Performs operation on collected data (sum, max, min, product, etc.)
- Computes and returns result
- Examples: Sum all values, find maximum, compute product

---

## **PART 2: Barrier Synchronization (Slides 9-11)**

### **MPI_Barrier Function (Slide 10)**

**Function Signature**:
```c
int MPI_Barrier(MPI_Comm comm);
```

**Parameters**:
- `comm` (IN): Communicator (handle)

**Behavior**:
```
Barrier Behavior:
├── Call returns at any process ONLY after ALL members have entered
├── Creates synchronization point across all processes
├── No data is transmitted (pure synchronization)
└── Equivalent to checkpoint that all must reach before proceeding
```

**Visual Representation**:
```
Before Barrier:
Process 0: ──────────────────────|MPI_Barrier()
Process 1: ──────|MPI_Barrier()  (waiting...)
Process 2: ───────────|MPI_Barrier() (waiting...)
Process 3: ────────────────|MPI_Barrier() (waiting...)

After All Arrive:
Process 0: ──────────────────────|───→ continues
Process 1: ──────|──────────────|───→ continues
Process 2: ───────────|─────────|───→ continues
Process 3: ────────────────|────|───→ continues
                             ↑
                    All released simultaneously
```

### **Barrier Synchronization Example (Slide 11)**

**Complete Working Code**:
```c
#include "stdio.h"
#include "string.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
    int comm_size;
    int my_rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // First phase: print 0-4
    for(int i=0; i<5; i++)
        printf("process %d: %d\n", my_rank, i);
    
    printf("waiting.....\n");
    
    // BARRIER: All processes must reach here before any continue
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Second phase: print 5-9 (only after ALL processes finished 0-4)
    for(int i=5; i<10; i++)
        printf("process %d: %d\n", my_rank, i);
    
    MPI_Finalize();
    return 0;
}
```

**What This Code Does**:
```
Step-by-Step Execution:

1. Each process prints numbers 0-4 independently
   - Processes may finish at different times
   - Output may be interleaved

2. Each process prints "waiting....."
   - First finisher waits for others at barrier

3. MPI_Barrier(MPI_COMM_WORLD) called
   - Process blocks until ALL processes reach this line
   - No process can proceed past barrier until all arrive

4. After barrier releases (all arrived):
   - All processes continue together
   - Each prints numbers 5-9

Result: GUARANTEES all processes finish printing 0-4 before 
        ANY process starts printing 5-9
```

**Example Output** (4 processes):
```
process 0: 0
process 0: 1
process 2: 0
process 1: 0
process 0: 2
process 3: 0
process 0: 3
process 1: 1
process 0: 4
process 2: 1
waiting.....
process 1: 2
waiting.....
process 2: 2
process 3: 1
waiting.....
process 3: 2
waiting.....
[All waiting at barrier until last process arrives]
process 0: 5
process 1: 5
process 2: 5
process 3: 5
[... rest of 5-9 for all processes ...]
```

**When to Use Barriers**:
```
Use MPI_Barrier when:
âœ" Need to ensure all processes complete a phase before starting next
âœ" Debugging parallel programs (force deterministic execution)
âœ" Timing measurements (synchronize before timing critical section)
âœ" Ensuring output ordering for presentation

Avoid MPI_Barrier when:
âœ— Unnecessary (adds synchronization overhead)
âœ— Better alternatives exist (collective operations have implicit sync)
âœ— Can cause performance degradation in well-designed algorithms
```

---

## **PART 3: Global Communication - Basic Patterns (Slides 12-14)**

### **Three Basic Patterns (Slide 13)**

**Pattern Classification**:
```
Pattern 1: Root sends to all processes (including itself)
├── MPI_Bcast - Broadcast same data to everyone
└── MPI_Scatter - Distribute different data portions to each process

Pattern 2: Root receives data from all processes (including itself)
└── MPI_Gather - Collect data from all processes to root

Pattern 3: Each process communicates with each process (including itself)
├── MPI_Allgather - All processes gather data from all
└── MPI_Alltoall - Complete data exchange between all processes
```

### **Global Communication Overview - THE MOST IMPORTANT SLIDE (Slide 14)**

**Visual Communication Patterns**:

This slide shows the fundamental data movement patterns. Understanding these is **critical** for the exam.

```
BROADCAST (One → All, same data):
Before:                    After:
P0 [A][ ][ ][ ]           P0 [A][ ][ ][ ]
P1 [ ][ ][ ][ ]           P1 [A][ ][ ][ ]
P2 [ ][ ][ ][ ]           P2 [A][ ][ ][ ]
P3 [ ][ ][ ][ ]           P3 [A][ ][ ][ ]

Key: Root (P0) sends SAME data (A) to all processes


SCATTER (One → All, different data):
Before:                    After:
P0 [A][B][C][D]           P0 [A][ ][ ][ ]
P1 [ ][ ][ ][ ]           P1 [B][ ][ ][ ]
P2 [ ][ ][ ][ ]           P2 [C][ ][ ][ ]
P3 [ ][ ][ ][ ]           P3 [D][ ][ ][ ]

Key: Root (P0) distributes DIFFERENT data to each process


GATHER (All → One, collecting data):
Before:                    After:
P0 [A][ ][ ][ ]           P0 [A][B][C][D]
P1 [B][ ][ ][ ]           P1 [B][ ][ ][ ]
P2 [C][ ][ ][ ]           P2 [C][ ][ ][ ]
P3 [D][ ][ ][ ]           P3 [D][ ][ ][ ]

Key: Root (P0) collects data from all processes


ALLGATHER (All → All, everyone gets everything):
Before:                    After:
P0 [A][ ][ ][ ]           P0 [A][B][C][D]
P1 [B][ ][ ][ ]           P1 [A][B][C][D]
P2 [C][ ][ ][ ]           P2 [A][B][C][D]
P3 [D][ ][ ][ ]           P3 [A][B][C][D]

Key: ALL processes get COMPLETE copy of everyone's data


ALLTOALL (All ↔ All, complete exchange):
Before:                    After:
P0 [A0][A1][A2][A3]       P0 [A0][B0][C0][D0]
P1 [B0][B1][B2][B3]       P1 [A1][B1][C1][D1]
P2 [C0][C1][C2][C3]       P2 [A2][B2][C2][D2]
P3 [D0][D1][D2][D3]       P3 [A3][B3][C3][D3]

Key: Think "transpose" - Each process sends different data to each other process
```

**Critical Understanding**:
- **Color coding in diagrams**: Same color = came from same source process
- **Order preservation**: Data arrives in rank order (deterministic)
- **Root concept**: Broadcast, Scatter, Gather have designated root; All* operations have no root

---

## **PART 4: Broadcast Communication (Slides 15-17)**

### **MPI_Bcast Overview (Slide 15)**

**What is Broadcast?**
- One of the **standard** collective communication techniques
- One process sends the **same data** to all processes in communicator
- Root process has initial copy, all others receive copy

**Common Use Cases**:
- Send user input to all parallel processes
- Distribute configuration parameters
- Share initial conditions for simulation
- Send command/control information

**Communication Tree Visualization**:
```
Process Rank 0 (root) broadcasts to all:

                    (0)
                   / | \
                  /  |  \
                 /   |   \
                /    |    \
              (1)   (2)  (3)  (4)  (5)  (6)  (7)

All processes receive identical copy of data from process 0
```

**Key Insight**: MPI implementation may use tree-based communication for efficiency rather than sequential sends.

### **MPI_Bcast API (Slide 16)**

**Function Signature**:
```c
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, 
              int root, MPI_Comm comm);
```

**Parameters**:
- `buffer` **(INOUT)**: Starting address of buffer
  - **Important**: INOUT because root writes, others read same buffer location
- `count` (IN): Number of elements in buffer
- `datatype` (IN): Datatype of the buffer elements
- `root` (IN): The rank of the root process in the communicator
- `comm` (IN): The communicator

**Critical Parameter Detail - INOUT Buffer**:
```
Why buffer is INOUT:

Root process (rank == root):
├── Buffer contains data to send (INPUT)
└── Data stays in buffer after broadcast

Non-root processes (rank != root):
├── Buffer receives data (OUTPUT)
└── Must have allocated buffer of correct size before calling

Example:
int value;
if (rank == 0) value = 42;  // Root sets value
MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Now ALL processes have value == 42
```

### **MPI_Bcast Example (Slide 17)**

**Complete Working Code**:
```c
#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
    int rank;
    int ibuf;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0)
        ibuf = 12345;
    else // set ibuf zero for non-root processes
        ibuf = 0;
    
    // Broadcast value from root (rank 0) to all processes
    MPI_Bcast(&ibuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0)
        printf("my rank = %d  ibuf = %d\n", rank, ibuf);
    
    MPI_Finalize();
    return 0;
}
```

**Step-by-Step Execution** (4 processes):

```
BEFORE MPI_Bcast:
┌──────────────────────────────────────────────┐
│ Rank 0: ibuf = 12345  (root has value)      │
│ Rank 1: ibuf = 0      (uninitialized)       │
│ Rank 2: ibuf = 0      (uninitialized)       │
│ Rank 3: ibuf = 0      (uninitialized)       │
└──────────────────────────────────────────────┘

DURING MPI_Bcast(&ibuf, 1, MPI_INT, 0, MPI_COMM_WORLD):
┌──────────────────────────────────────────────┐
│ Step 1: Root (rank 0) prepares data         │
│ Step 2: MPI sends 12345 from rank 0 to all  │
│ Step 3: Each process receives into ibuf     │
└──────────────────────────────────────────────┘

AFTER MPI_Bcast:
┌──────────────────────────────────────────────┐
│ Rank 0: ibuf = 12345  (unchanged)           │
│ Rank 1: ibuf = 12345  (received from root)  │
│ Rank 2: ibuf = 12345  (received from root)  │
│ Rank 3: ibuf = 12345  (received from root)  │
└──────────────────────────────────────────────┘

OUTPUT:
my rank = 1  ibuf = 12345
my rank = 2  ibuf = 12345
my rank = 3  ibuf = 12345
```

**Why Rank 0 Doesn't Print**: Code has `if (rank != 0)` - root already knows value, only non-root processes confirm receipt.

**Common Mistakes to Avoid**:
```
âœ— Forgetting to allocate buffer on non-root processes
âœ— Using different count/datatype on different processes
âœ— Not calling on ALL processes in communicator
âœ— Assuming buffer starts empty on root (it contains data to send!)
```

---

## **PART 5: Scatter and Gather Communication (Slides 18-21)**

### **Scatter and Gather Overview (Slide 18)**

**Visual Comparison**:

```
MPI_Bcast (Review):
Takes SINGLE data element at root → copies to all

        Root: [■]
               ↓
        ┌──────┼──────┐
        ↓      ↓      ↓
    P0: [■] P1: [■] P2: [■] P3: [■]
    
    All processes get IDENTICAL data


MPI_Scatter:
Takes ARRAY of elements → distributes in rank order

        Root: [■][■][■][■]
               ↓  ↓  ↓  ↓
              ┌───┼──┼──┐
              ↓   ↓  ↓  ↓
    P0: [■] P1: [■] P2: [■] P3: [■]
    
    Each process gets DIFFERENT portion


MPI_Gather (inverse of Scatter):
Collects elements from all → assembles array at root

    P0: [■] P1: [■] P2: [■] P3: [■]
         ↓   ↓   ↓   ↓
         └───┼───┼───┘
             ↓
        Root: [■][■][■][■]
    
    Root assembles array in rank order
```

**Key Principles**:
- **Order matters**: Distribution/collection follows rank order
- **Deterministic**: Always same ordering (reproducible)
- **Symmetric operations**: Scatter and Gather are inverses

**Communication Tree Visualization**:
```
MPI_Bcast: One-to-many, same data
                (0)■
               / | \
              /  |  \
            (1)■ (2)■ (3)■


MPI_Scatter: One-to-many, different data
                (0)[A][B][C]
                / |  \
               /  |   \
            (0)A (1)B (2)C


MPI_Gather: Many-to-one, collecting data
            (0)A (1)B (2)C
               \ | /
                \|/
              (0)[A][B][C]
```

### **MPI_Scatter API (Slide 19)**

**Function Signature**:
```c
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                int root, MPI_Comm comm);
```

**Input Parameters**:
- `sendbuf` (IN): Address of send buffer (choice, **significant only at root**)
- `sendcount` (IN): Number of elements sent to **each process** (integer, **significant only at root**)
- `sendtype` (IN): Data type of send buffer elements (**significant only at root**) (handle)
- `recvcount` (IN): Number of elements in receive buffer (integer)
- `recvtype` (IN): Data type of receive buffer elements (handle)
- `root` (IN): Rank of sending process (integer)
- `comm` (IN): Communicator (handle)

**Output Parameters**:
- `recvbuf` (OUT): Address of receive buffer

**Critical Understanding - "Significant Only at Root"**:
```
What this means:

At ROOT process:
├── sendbuf: Must point to array with sendcount × numprocs elements
├── sendcount: Number of elements to send to EACH process
├── sendtype: Type of elements in sendbuf
└── These parameters control what gets scattered

At NON-ROOT processes:
├── sendbuf: Can be NULL or anything (ignored)
├── sendcount: Ignored
├── sendtype: Ignored
└── Only recvbuf, recvcount, recvtype matter

Example:
float data[100];  // Only root needs this
float my_portion[25];
if (rank == 0) {
    // Initialize data[100]
}
MPI_Scatter(data, 25, MPI_FLOAT,  // Only meaningful on root
            my_portion, 25, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

**Visual Parameter Explanation** (4 processes):
```
Root process (rank 0):
┌────────────────────────────────────────┐
│ sendbuf: [0][1][2][3][4][5][6][7]     │ ← Total array
│           └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘     │
│             │     │     │     │       │
│        sendcount=2 for each process   │
└────────────────────────────────────────┘

Distribution:
P0 receives: [0][1]  (first sendcount elements)
P1 receives: [2][3]  (next sendcount elements)
P2 receives: [4][5]  (next sendcount elements)
P3 receives: [6][7]  (last sendcount elements)

Each process recvbuf gets exactly sendcount elements
```

### **MPI_Gather API (Slide 20)**

**Function Signature**:
```c
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
```

**Input Parameters**:
- `sendbuf` (IN): Starting address of send buffer (choice)
- `sendcount` (IN): Number of elements in send buffer (integer)
- `sendtype` (IN): Data type of send buffer elements (handle)
- `recvcount` (IN): Number of elements for **any single receive** (integer, **significant only at root**)
- `recvtype` (IN): Data type of recv buffer elements (**significant only at root**) (handle)
- `root` (IN): Rank of receiving process (integer)
- `comm` (IN): Communicator (handle)

**Output Parameters**:
- `recvbuf` (OUT): Address of receive buffer (choice, **significant only at root**)

**Critical Understanding - Gather Sizing**:
```
ALL processes:
├── sendbuf: Must contain sendcount elements
├── sendcount: How many elements this process sends
├── All processes send SAME number of elements
└── Each process participates in gather

ROOT process only:
├── recvbuf: Must be large enough for recvcount × numprocs elements
├── recvcount: How many elements to receive from EACH process
├── recvcount should equal sendcount
└── Assembles data in rank order

Example (4 processes):
int my_data[10];      // Each process has 10 elements
int all_data[40];     // Root needs 10 × 4 = 40 space

// Each process sends 10, root receives 10 from each
MPI_Gather(my_data, 10, MPI_INT,
           all_data, 10, MPI_INT, 0, MPI_COMM_WORLD);
```

**Visual Gather Process** (4 processes):
```
BEFORE Gather:
P0: [A0][A1]  (sendcount=2)
P1: [B0][B1]  (sendcount=2)
P2: [C0][C1]  (sendcount=2)
P3: [D0][D1]  (sendcount=2)

Root (P0) has recvbuf allocated for 2×4=8 elements

DURING Gather:
Each process sends its sendbuf contents
Root collects in rank order

AFTER Gather:
P0: [A0][A1]                              (original data)
P1: [B0][B1]                              (original data)
P2: [C0][C1]                              (original data)
P3: [D0][D1]                              (original data)

Root's recvbuf: [A0][A1][B0][B1][C0][C1][D0][D1]
                 └─P0─┘ └─P1─┘ └─P2─┘ └─P3─┘
```

### **Scatter and Gather Example - Computing Average (Slide 21)**

**Complete Working Code**:
```c
// This example shows using both Scatter and Gather to compute average

float *rand_nums = NULL;
if (world_rank == 0) {
    rand_nums = create_rand_nums(elements_per_proc * world_size);
}

// Create a buffer that will hold a subset of the random numbers
float *sub_rand_nums = malloc(sizeof(float) * elements_per_proc);

// Scatter the random numbers to all processes
MPI_Scatter(rand_nums, elements_per_proc, MPI_FLOAT, 
            sub_rand_nums, elements_per_proc, MPI_FLOAT, 
            0, MPI_COMM_WORLD);

// Compute the average of your subset
float sub_avg = compute_avg(sub_rand_nums, elements_per_proc);

// Gather all partial averages down to the root process
float *sub_avgs = NULL;
if (world_rank == 0) {
    sub_avgs = malloc(sizeof(float) * world_size);
}

MPI_Gather(&sub_avg, 1, MPI_FLOAT, 
           sub_avgs, 1, MPI_FLOAT, 
           0, MPI_COMM_WORLD);

// Compute the total average of all numbers.
if (world_rank == 0) {
    float avg = compute_avg(sub_avgs, world_size);
}
```

**Step-by-Step Execution Analysis** (4 processes, 16 elements total):

```
STEP 1: Root Creates Data
┌─────────────────────────────────────────────────────────────┐
│ Rank 0 (root): rand_nums[16] = [3.2, 1.7, 4.8, ...]         │
│ Rank 1-3: rand_nums = NULL (don't need full array)          │
└─────────────────────────────────────────────────────────────┘

STEP 2: Scatter Distributes Data (elements_per_proc = 4)
┌─────────────────────────────────────────────────────────────┐
│ MPI_Scatter sends 4 elements to each process:               │
│                                                             │
│ Rank 0 gets: sub_rand_nums[4] = [3.2, 1.7, 4.8, 2.1]        │
│ Rank 1 gets: sub_rand_nums[4] = [5.3, 0.9, 3.4, 6.2]        │
│ Rank 2 gets: sub_rand_nums[4] = [2.8, 4.1, 1.5, 3.9]        │
│ Rank 3 gets: sub_rand_nums[4] = [4.7, 2.3, 5.8, 1.2]        │
└─────────────────────────────────────────────────────────────┘

STEP 3: Each Process Computes Local Average
┌─────────────────────────────────────────────────────────────┐
│ Rank 0: sub_avg = (3.2+1.7+4.8+2.1)/4 = 2.95              │
│ Rank 1: sub_avg = (5.3+0.9+3.4+6.2)/4 = 3.95              │
│ Rank 2: sub_avg = (2.8+4.1+1.5+3.9)/4 = 3.08              │
│ Rank 3: sub_avg = (4.7+2.3+5.8+1.2)/4 = 3.50              │
└─────────────────────────────────────────────────────────────┘

STEP 4: Gather Collects Partial Averages
┌─────────────────────────────────────────────────────────────┐
│ MPI_Gather collects one float from each process:           │
│                                                             │
│ Rank 0 sends: 2.95                                         │
│ Rank 1 sends: 3.95                                         │
│ Rank 2 sends: 3.08                                         │
│ Rank 3 sends: 3.50                                         │
│                                                             │
│ Root receives: sub_avgs[4] = [2.95, 3.95, 3.08, 3.50]     │
└─────────────────────────────────────────────────────────────┘

STEP 5: Root Computes Final Average
┌─────────────────────────────────────────────────────────────┐
│ avg = (2.95 + 3.95 + 3.08 + 3.50) / 4 = 3.37              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insights**:
```
Why this pattern works:
âœ" Distributes work evenly (each process handles elements_per_proc items)
âœ" Minimizes communication (only 2 collective calls)
âœ" Scales well (more processes = less work per process)
âœ" Clean code (no complex send/recv logic)

Performance considerations:
├── Scatter communication time: O(n) where n = total elements
├── Computation time per process: O(n/p) where p = num processes
├── Gather communication time: O(p)
└── Total: O(n/p) computation, O(n+p) communication
```

---

## **PART 6: Allgather Communication (Slides 22-23)**

### **MPI_Allgather Overview (Slide 22)**

**Definition**: Given a set of elements distributed across all processes, MPI_Allgather will gather all of the elements to **all the processes**.

**Key Difference from MPI_Gather**:
```
MPI_Gather:
├── Collects data from all processes
├── Only ROOT gets complete collection
└── Non-root processes keep only their original data

MPI_Allgather:
├── Collects data from all processes
├── ALL processes get complete collection
└── Every process ends with same complete data set
```

**Visual Representation**:
```
BEFORE Allgather:
P0: [RED][ ][ ][ ]
P1: [GRN][ ][ ][ ]
P2: [BLU][ ][ ][ ]
P3: [YLW][ ][ ][ ]

Each process has ONE element (its own data)

AFTER Allgather:
P0: [RED][GRN][BLU][YLW]
P1: [RED][GRN][BLU][YLW]
P2: [RED][GRN][BLU][YLW]
P3: [RED][GRN][BLU][YLW]

ALL processes now have COMPLETE collection
```

**Communication Pattern**:
```
MPI_Allgather visualization:

      P0: [■]      P1: [■]      P2: [■]
         ↓ ↘       ↓ ↕ ↙       ↓ ↙
         ↓   ↘     ↓ ↕ ↙       ↓
         ↓     ↘   ↓ ↕ ↙       ↓
         ↓       ↘ ↓ ↕ ↙       ↓
         ↓         ↓ ↕         ↓
      [■][■][■] [■][■][■] [■][■][■]
      
All processes exchange data with all others
```

**Equivalent Operation**:
```c
// MPI_Allgather is equivalent to:
MPI_Gather(..., root, ...);  // Gather to root
MPI_Bcast(..., root, ...);   // Broadcast from root

// But MPI_Allgather is:
// ✓ More efficient (optimized communication pattern)
// ✓ Single function call
// ✓ No explicit root needed
```

### **MPI_Allgather Example - Computing Average (Slide 23)**

**Complete Working Code**:
```c
// Gather all partial averages down to all the processes
float *sub_avgs = (float *)malloc(sizeof(float) * world_size);

MPI_Allgather(&sub_avg, 1, MPI_FLOAT, 
              sub_avgs, 1, MPI_FLOAT, 
              MPI_COMM_WORLD);

// Compute the total average of all numbers.
float avg = compute_avg(sub_avgs, world_size);
```

**Function Signature**:
```c
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                  MPI_Comm comm);
```

**Parameters**:
- `sendbuf` (IN): Starting address of send buffer
- `sendcount` (IN): Number of elements in send buffer
- `sendtype` (IN): Datatype of send buffer elements
- `recvbuf` (OUT): Address of receive buffer
- `recvcount` (IN): Number of elements received from **any single process**
- `recvtype` (IN): Datatype of receive buffer elements
- `comm` (IN): Communicator

**Step-by-Step Execution** (4 processes):

```
STEP 1: Each Process Has Computed Its Partial Average
┌─────────────────────────────────────────────────────────────┐
│ Rank 0: sub_avg = 2.95                                      │
│ Rank 1: sub_avg = 3.95                                      │
│ Rank 2: sub_avg = 3.08                                      │
│ Rank 3: sub_avg = 3.50                                      │
└─────────────────────────────────────────────────────────────┘

STEP 2: Allgather Collects and Distributes to All
┌─────────────────────────────────────────────────────────────┐
│ MPI_Allgather(&sub_avg, 1, MPI_FLOAT,                      │
│               sub_avgs, 1, MPI_FLOAT,                       │
│               MPI_COMM_WORLD);                              │
│                                                             │
│ Each process sends: 1 float (its sub_avg)                  │
│ Each process receives: 1 float from each of 4 processes    │
└─────────────────────────────────────────────────────────────┘

STEP 3: All Processes Have Complete Array
┌─────────────────────────────────────────────────────────────┐
│ Rank 0: sub_avgs[4] = [2.95, 3.95, 3.08, 3.50]            │
│ Rank 1: sub_avgs[4] = [2.95, 3.95, 3.08, 3.50]            │
│ Rank 2: sub_avgs[4] = [2.95, 3.95, 3.08, 3.50]            │
│ Rank 3: sub_avgs[4] = [2.95, 3.95, 3.08, 3.50]            │
│                                                             │
│ EVERY process has IDENTICAL copy                           │
└─────────────────────────────────────────────────────────────┘

STEP 4: All Processes Can Compute Final Average
┌─────────────────────────────────────────────────────────────┐
│ ALL processes can now compute:                             │
│ avg = compute_avg(sub_avgs, world_size)                    │
│ avg = (2.95 + 3.95 + 3.08 + 3.50) / 4 = 3.37              │
│                                                             │
│ No need for final Bcast - everyone already has result!     │
└─────────────────────────────────────────────────────────────┘
```

**Comparison: Gather vs Allgather**:

```
Using MPI_Gather (previous example):
├── Step 1: Scatter data
├── Step 2: Compute partial averages
├── Step 3: MPI_Gather to root
├── Step 4: Root computes final average
└── Result: Only root has final answer

Using MPI_Allgather (this example):
├── Step 1: Scatter data
├── Step 2: Compute partial averages
├── Step 3: MPI_Allgather to all
├── Step 4: ALL processes compute final average
└── Result: All processes have final answer

When to use each:
✓ Use Gather when only root needs result
✓ Use Allgather when all processes need result
```

---

## **PART 7: Alltoall Communication (Slide 24)**

### **MPI_Alltoall Overview**

**Definition**: Scatter data from all tasks to all tasks - a complete exchange where each process sends different data to each other process.

**Function Signature**:
```c
sendcnt = 1;
recvcnt = 1;
MPI_Alltoall(sendbuf, sendcnt, MPI_INT,
             recvbuf, recvcnt, MPI_INT,
             MPI_COMM_WORLD);
```

**Visual Representation - The "Transpose" Operation**:

```
BEFORE Alltoall (each row is one process's sendbuf):
┌─────────────────────────────────────────────┐
│ P0: [ 1 ][ 5 ][ 9 ][13]                    │
│ P1: [ 2 ][ 6 ][10 ][14]                    │
│ P2: [ 3 ][ 7 ][11 ][14]                    │
│ P3: [ 4 ][ 8 ][12 ][16]                    │
└─────────────────────────────────────────────┘

Think of this as a 4×4 matrix (sendbuf[4][4])

AFTER Alltoall (each row is one process's recvbuf):
┌─────────────────────────────────────────────┐
│ P0: [ 1 ][ 2 ][ 3 ][ 4]                    │
│ P1: [ 5 ][ 6 ][ 7 ][ 8]                    │
│ P2: [ 9 ][10 ][11 ][12]                    │
│ P3: [13 ][14 ][15 ][16]                    │
└─────────────────────────────────────────────┘

The matrix has been TRANSPOSED!
```

**Detailed Data Movement**:

```
What each process sends:
┌──────────────────────────────────────────────────────────┐
│ P0 sends: element [0] to P0, [1] to P1, [2] to P2, [3] to P3  │
│ P1 sends: element [0] to P0, [1] to P1, [2] to P2, [3] to P3  │
│ P2 sends: element [0] to P0, [1] to P1, [2] to P2, [3] to P3  │
│ P3 sends: element [0] to P0, [1] to P1, [2] to P2, [3] to P3  │
└──────────────────────────────────────────────────────────┘

What each process receives:
┌──────────────────────────────────────────────────────────┐
│ P0 receives: element from P0[0], P1[0], P2[0], P3[0]    │
│ P1 receives: element from P0[1], P1[1], P2[1], P3[1]    │
│ P2 receives: element from P0[2], P1[2], P2[2], P3[2]    │
│ P3 receives: element from P0[3], P1[3], P2[3], P3[3]    │
└──────────────────────────────────────────────────────────┘
```

**Code Example from Slide**:

```c
sendcnt = 1;
recvcnt = 1;
MPI_Alltoall(sendbuf, sendcnt, MPI_INT,
             recvbuf, recvcnt, MPI_INT,
             MPI_COMM_WORLD);
```

**Execution with Real Data** (4 processes):

```
Initial State:
┌────────────────────────────────────────────────────────────┐
│ Rank 0 sendbuf[4] = [1, 5, 9, 13]                         │
│ Rank 1 sendbuf[4] = [2, 6, 10, 14]                        │
│ Rank 2 sendbuf[4] = [3, 7, 11, 14]   (note: should be 15) │
│ Rank 3 sendbuf[4] = [4, 8, 12, 16]                        │
└────────────────────────────────────────────────────────────┘

Communication Pattern (sendcnt=1, recvcnt=1):
┌────────────────────────────────────────────────────────────┐
│ P0 sends sendbuf[0]=1  to P0                               │
│ P0 sends sendbuf[1]=5  to P1                               │
│ P0 sends sendbuf[2]=9  to P2                               │
│ P0 sends sendbuf[3]=13 to P3                               │
│                                                            │
│ P1 sends sendbuf[0]=2  to P0                               │
│ P1 sends sendbuf[1]=6  to P1                               │
│ ... (pattern continues for P2 and P3)                      │
└────────────────────────────────────────────────────────────┘

Final State (after transpose):
┌────────────────────────────────────────────────────────────┐
│ Rank 0 recvbuf[4] = [1, 2, 3, 4]    (column 0)            │
│ Rank 1 recvbuf[4] = [5, 6, 7, 8]    (column 1)            │
│ Rank 2 recvbuf[4] = [9, 10, 11, 12] (column 2)            │
│ Rank 3 recvbuf[4] = [13, 14, 15, 16] (column 3)           │
└────────────────────────────────────────────────────────────┘
```

**Use Cases**:
```
Matrix Transpose:
├── Distributed matrix stored row-wise
├── Need to access column-wise
└── Alltoall performs transpose efficiently

Fast Fourier Transform (FFT):
├── Parallel FFT requires data reorganization
├── Alltoall handles data redistribution
└── Critical for scientific computing

General Data Redistribution:
├── Change from one distribution pattern to another
├── Example: Row-major to column-major
└── All processes need data from all others
```

---

## **PART 8: Global Reductions (Slides 25-34)**

### **Global Reductions Overview (Slide 26)**

**Definition**: Global reductions perform some numerical operation in a distributed manner and are extremely useful in many cases.

**Key Concepts**:
```
Global Reductions:
├── Analogous to reduction operators in OpenMP
├── Perform operation across all processes
├── Common operations: Max, Min, Sum, Product
├── Can replace send/recv with broadcast/reduce patterns
└── Fundamental to many numerical algorithms
```

**Common Operations Available** (from Slide 28):
```
Arithmetic Operations:
├── MPI_MAX - Returns maximum element
├── MPI_MIN - Returns minimum element
├── MPI_SUM - Sums the elements
└── MPI_PROD - Multiplies all elements

Logical Operations:
├── MPI_LAND - Logical AND across elements
└── MPI_LOR - Logical OR across elements

Bitwise Operations:
├── MPI_BAND - Bitwise AND across bits of elements
└── MPI_BOR - Bitwise OR across bits of elements

Location Operations:
├── MPI_MAXLOC - Returns maximum value and rank of process that owns it
└── MPI_MINLOC - Returns minimum value and rank of process that owns it
```

### **MPI_Reduce Function (Slide 27)**

**Function Signature**:
```c
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, 
               MPI_Datatype datatype, MPI_Op op, 
               int root, MPI_Comm comm);
```

**Parameters**:
- `sendbuf` (IN): Address of send buffer
- `recvbuf` (OUT): Address of receive buffer (significant only at root)
- `count` (IN): Number of elements in send buffer
- `datatype` (IN): Datatype of elements in buffer
- `op` (IN): **NEW** The reduce operation (MPI_SUM, MPI_MAX, etc.)
- `root` (IN): Rank of root process
- `comm` (IN): Communicator

**Visual Operation**:

```
BEFORE MPI_Reduce:
┌─────────────────────────────────────────────┐
│ Process 1:  sendbuf = [1]                   │
│ Process 2:  sendbuf = [2]                   │
│ Process 3:  sendbuf = [3]                   │
│ Process 4:  sendbuf = [4]                   │
└─────────────────────────────────────────────┘

MPI_Reduce with op=MPI_SUM, root=1:
┌─────────────────────────────────────────────┐
│ Performs: 1 + 2 + 3 + 4 = 10                │
└─────────────────────────────────────────────┘

AFTER MPI_Reduce:
┌─────────────────────────────────────────────┐
│ Process 1:  recvbuf = [10]  ← ROOT HAS SUM │
│ Process 2:  recvbuf = [ ]   (undefined)    │
│ Process 3:  recvbuf = [ ]   (undefined)    │
│ Process 4:  recvbuf = [ ]   (undefined)    │
└─────────────────────────────────────────────┘

Only root process receives the result!
```

**Tree-Based Reduction** (efficient implementation):
```
Parallel reduction tree for MPI_SUM:

Level 0:    [1]      [2]      [3]      [4]
              \      /          \      /
               \    /            \    /
Level 1:        [3]               [7]
                  \               /
                   \             /
Level 2:            [10] ← Root receives final sum

Time complexity: O(log n) instead of O(n) for sequential
```

### **MPI_Reduce Example - Computing Average (Slide 29)**

**Complete Working Code**:
```c
float *rand_nums = NULL;
rand_nums = create_rand_nums(num_elements_per_proc);

// Sum the numbers locally
float local_sum = 0;
int i;
for (i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
}

// Print the random numbers on each process
printf("Local sum for process %d - %f, avg = %f\n", 
       world_rank, local_sum, 
       local_sum / num_elements_per_proc);

// Reduce all of the local sums into the global sum
float global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
           MPI_COMM_WORLD);

// Print the result
if (world_rank == 0) {
    printf("Total sum = %f, avg = %f\n", global_sum,
           global_sum / (world_size * num_elements_per_proc));
}
```

**Step-by-Step Execution** (4 processes, 100 elements each):

```
STEP 1: Each Process Generates Random Numbers
┌──────────────────────────────────────────────────────────────┐
│ Rank 0: rand_nums[100] = [2.1, 4.5, 1.8, ...]               │
│ Rank 1: rand_nums[100] = [3.4, 0.9, 5.2, ...]               │
│ Rank 2: rand_nums[100] = [1.7, 3.8, 2.4, ...]               │
│ Rank 3: rand_nums[100] = [4.2, 1.1, 3.9, ...]               │
└──────────────────────────────────────────────────────────────┘

STEP 2: Each Process Computes Local Sum
┌──────────────────────────────────────────────────────────────┐
│ Rank 0: local_sum = 0                                        │
│         for (i=0; i<100; i++) local_sum += rand_nums[i]      │
│         local_sum = 245.8, avg = 245.8/100 = 2.458          │
│                                                              │
│ Rank 1: local_sum = 287.3, avg = 2.873                      │
│ Rank 2: local_sum = 312.5, avg = 3.125                      │
│ Rank 3: local_sum = 254.9, avg = 2.549                      │
└──────────────────────────────────────────────────────────────┘

Output from each process:
Local sum for process 0 - 245.800000, avg = 2.458000
Local sum for process 1 - 287.300000, avg = 2.873000
Local sum for process 2 - 312.500000, avg = 3.125000
Local sum for process 3 - 254.900000, avg = 2.549000

STEP 3: MPI_Reduce Sums All Local Sums
┌──────────────────────────────────────────────────────────────┐
│ MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT,           │
│            MPI_SUM, 0, MPI_COMM_WORLD);                      │
│                                                              │
│ Operation: 245.8 + 287.3 + 312.5 + 254.9 = 1100.5           │
│                                                              │
│ Result stored in global_sum on rank 0 only                  │
└──────────────────────────────────────────────────────────────┘

STEP 4: Root Computes and Prints Global Average
┌──────────────────────────────────────────────────────────────┐
│ if (world_rank == 0) {                                       │
│     avg = global_sum / (world_size * num_elements_per_proc)  │
│     avg = 1100.5 / (4 * 100) = 1100.5 / 400 = 2.75125      │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘

Output from root:
Total sum = 1100.500000, avg = 2.751250
```

**Why This is Efficient**:
```
Comparison with Gather approach:

Using MPI_Gather:
├── Transfer all data to root: 4 × 100 floats = 400 floats
├── Root computes sum: 400 additions
└── Communication: O(n) where n = total elements

Using MPI_Reduce:
├── Each process computes local sum: 100 additions each
├── Transfer only local sums: 4 floats
├── MPI_Reduce combines: O(log p) communication rounds
└── Communication: O(log p) where p = num processes

Reduction wins:
✓ Less data transfer (4 floats vs 400 floats)
✓ Parallel computation (all processes compute local sums)
✓ Logarithmic communication pattern
✓ Scales better with more processes
```

### **MPI_Allreduce Function (Slide 30)**

**Function Signature**:
```c
int MPI_Allreduce(void* send_data, void* recv_data, int count,
                  MPI_Datatype datatype, MPI_Op op, 
                  MPI_Comm communicator);
```

**Definition**: Combines elements in all sendbufs of each process (using an operation) and returns that value to **all processes**.

**Visual Representation**:

```
Data organized as matrix (each row = one process):

BEFORE Allreduce (+ operation):
┌──────────────────────────────────────────────┐
│      Data                                    │
│ P0:  [A₀][B₀][C₀][D₀]                       │
│ P1:  [A₁][B₁][C₁][D₁]                       │
│ P2:  [A₂][B₂][C₂][D₂]                       │
│ P3:  [A₃][B₃][C₃][D₃]                       │
└──────────────────────────────────────────────┘

AFTER Allreduce (+ operation):
┌──────────────────────────────────────────────────────────────┐
│ P0:  [A₀+A₁+A₂+A₃][B₀+B₁+B₂+B₃][C₀+C₁+C₂+C₃][D₀+D₁+D₂+D₃]  │
│ P1:  [A₀+A₁+A₂+A₃][B₀+B₁+B₂+B₃][C₀+C₁+C₂+C₃][D₀+D₁+D₂+D₃]  │
│ P2:  [A₀+A₁+A₂+A₃][B₀+B₁+B₂+B₃][C₀+C₁+C₂+C₃][D₀+D₁+D₂+D₃]  │
│ P3:  [A₀+A₁+A₂+A₃][B₀+B₁+B₂+B₃][C₀+C₁+C₂+C₃][D₀+D₁+D₂+D₃]  │
└──────────────────────────────────────────────────────────────┘

Key: ALL processes receive IDENTICAL result array
```

**Column-wise Operation**:
```
Think of it as reducing each column:

Column 0: A₀+A₁+A₂+A₃ → all processes get this in position 0
Column 1: B₀+B₁+B₂+B₃ → all processes get this in position 1
Column 2: C₀+C₁+C₂+C₃ → all processes get this in position 2
Column 3: D₀+D₁+D₂+D₃ → all processes get this in position 3
```

**Comparison: Reduce vs Allreduce**:
```
MPI_Reduce:
├── Performs reduction operation
├── Only ROOT receives result
├── Other processes don't get result
└── Use when only one process needs answer

MPI_Allreduce:
├── Performs reduction operation
├── ALL processes receive result
├── Everyone gets same answer
└── Use when all processes need answer

Equivalence:
MPI_Allreduce(...) ≈ MPI_Reduce(...) + MPI_Bcast(...)
(But Allreduce is optimized internally)
```

### **MPI_Scan Function (Slide 31)**

**Function Signature**:
```c
int MPI_Scan(void *sendbuf, void *recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
```

**Definition**: Combines elements in all sendbufs of each process and the 'prior' result - performs a **prefix sum** (or prefix operation).

**Visual Representation**:

```
BEFORE Scan (+ operation):
┌──────────────────────────────────────────────┐
│      Data                                    │
│ P0:  [A₀][B₀][C₀][D₀]                       │
│ P1:  [A₁][B₁][C₁][D₁]                       │
│ P2:  [A₂][B₂][C₂][D₂]                       │
│ P3:  [A₃][B₃][C₃][D₃]                       │
└──────────────────────────────────────────────┘

AFTER Scan (+ operation):
┌─────────────────────────────────────────────────────────────────┐
│ P0:  [A₀]           [B₀]           [C₀]           [D₀]          │
│ P1:  [A₀+A₁]        [B₀+B₁]        [C₀+C₁]        [D₀+D₁]       │
│ P2:  [A₀+A₁+A₂]     [B₀+B₁+B₂]     [C₀+C₁+C₂]     [D₀+D₁+D₂]    │
│ P3:  [A₀+A₁+A₂+A₃]  [B₀+B₁+B₂+B₃]  [C₀+C₁+C₂+C₃]  [D₀+D₁+D₂+D₃] │
└─────────────────────────────────────────────────────────────────┘

Key: Each process gets CUMULATIVE result up to and including its rank
```

**Concrete Example** (integers with MPI_SUM):

```
BEFORE Scan:
P0: [5][10][15][20]
P1: [2][ 4][ 6][ 8]
P2: [1][ 3][ 5][ 7]
P3: [3][ 6][ 9][12]

AFTER Scan (cumulative sum by column):
P0: [ 5][10][15][20]           ← Just P0's values
P1: [ 7][14][21][28]           ← P0 + P1
P2: [ 8][17][26][35]           ← P0 + P1 + P2
P3: [11][23][35][47]           ← P0 + P1 + P2 + P3

Column 0: P0=5, P1=5+2=7, P2=5+2+1=8, P3=5+2+1+3=11
Column 1: P0=10, P1=10+4=14, P2=10+4+3=17, P3=10+4+3+6=23
... and so on
```

**Use Cases for Scan**:
```
Prefix Sum Applications:
├── Computing cumulative totals
├── Parallel array indexing (determine where to write data)
├── Load balancing (cumulative work assignments)
└── Computational geometry algorithms

Example - Parallel Array Construction:
Each process knows: "I need to write 10 elements"
After scan: Each knows: "I start writing at position X"
```

### **MPI_Reduce_scatter Function (Slide 32)**

**Function Signature**:
```c
int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, 
                       const int recvcounts[], 
                       MPI_Datatype datatype, MPI_Op op, 
                       MPI_Comm comm);
```

**Definition**: Combines elements in all sendbufs in chunks of size *n* of each process (using an operation), then distributes the resulting array over *n* processes.

**Visual Representation**:

```
BEFORE Reduce_scatter (+ operation):
┌──────────────────────────────────────────────┐
│      Data (each process has 4 elements)      │
│ P0:  [A₀][B₀][C₀][D₀]                       │
│ P1:  [A₁][B₁][C₁][D₁]                       │
│ P2:  [A₂][B₂][C₂][D₂]                       │
│ P3:  [A₃][B₃][C₃][D₃]                       │
└──────────────────────────────────────────────┘

AFTER Reduce_scatter (+ operation):
┌──────────────────────────────────────────────┐
│ P0:  [A₀+A₁+A₂+A₃][ ][ ][ ]                 │
│ P1:  [ ][B₀+B₁+B₂+B₃][ ][ ]                 │
│ P2:  [ ][ ][C₀+C₁+C₂+C₃][ ]                 │
│ P3:  [ ][ ][ ][D₀+D₁+D₂+D₃]                 │
└──────────────────────────────────────────────┘

Key: Each process gets ONE reduced chunk
```

**What Reduce_scatter Does**:
```
Step 1: Perform reduction on each column (like Allreduce)
├── Column 0: A₀+A₁+A₂+A₃
├── Column 1: B₀+B₁+B₂+B₃
├── Column 2: C₀+C₁+C₂+C₃
└── Column 3: D₀+D₁+D₂+D₃

Step 2: Scatter results to processes
├── P0 gets reduced column 0
├── P1 gets reduced column 1
├── P2 gets reduced column 2
└── P3 gets reduced column 3

Equivalent to: MPI_Allreduce + MPI_Scatter
(But more efficient when combined)
```

**Use Case Example**:
```c
// Each process has computed partial results for all processes
// Want to reduce and distribute in one operation

float my_contributions[4] = {1.0, 2.0, 3.0, 4.0};
float my_result;
int recvcounts[4] = {1, 1, 1, 1}; // Each process gets 1 element

MPI_Reduce_scatter(my_contributions, &my_result, recvcounts,
                   MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

// Each process now has sum of all contributions for its portion
```

### **Custom Reduction Operations (Slides 33-34)**

**Overview (Slide 33)**:

You can define your own reduction operation, with requirements:
- **Must be associative**: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
- **Can be commutative** (optional, specified when creating)
- **No MPI communication** inside the custom reduction function

**Examples of Associative Operations**:
```
Associative (Valid):
✓ Max: max(max(a,b),c) = max(a,max(b,c))
✓ Min: min(min(a,b),c) = min(a,min(b,c))
✓ Sum: (a+b)+c = a+(b+c)
✓ Product: (a×b)×c = a×(b×c)
✓ Absolute maximum: max(|a|,|b|,|c|)

Not Associative (Invalid):
✗ Subtraction: (a-b)-c ≠ a-(b-c)
✗ Division: (a/b)/c ≠ a/(b/c)
✗ Exponentiation: (a^b)^c ≠ a^(b^c)
```

**Creating Custom Reduction (Slide 34)**:

**Function Signatures**:
```c
// Step 1: Define your reduction function
typedef void MPI_User_function(void *invec, void *inoutvec, 
                                int *len, MPI_Datatype *datatype);

// Step 2: Create the operation handle
int MPI_Op_create(MPI_User_function *function, int commute, 
                  MPI_Op *op);
```

**Parameters for MPI_Op_create**:
- `function` (IN): The user-defined function
- `commute` (IN): True (1) if commutative, false (0) otherwise
- `op` (OUT): The operation handle

**Complete Example - Absolute Maximum**:

```c
// Step 1: Define the custom reduction function
void abs_max_function(void *invec, void *inoutvec, int *len, 
                      MPI_Datatype *datatype)
{
    int i;
    float *in = (float *)invec;
    float *inout = (float *)inoutvec;
    
    for (i = 0; i < *len; i++) {
        float abs_in = (in[i] < 0) ? -in[i] : in[i];
        float abs_inout = (inout[i] < 0) ? -inout[i] : inout[i];
        
        if (abs_in > abs_inout) {
            inout[i] = in[i]; // Keep original sign, but larger absolute value
        }
    }
}

// Step 2: Create the MPI operation
MPI_Op abs_max_op;
MPI_Op_create(abs_max_function, 1, &abs_max_op); // 1 = commutative

// Step 3: Use in MPI_Reduce
float my_value = -5.0; // Process has value -5.0
float result;

MPI_Reduce(&my_value, &result, 1, MPI_FLOAT, abs_max_op, 
           0, MPI_COMM_WORLD);

// If processes have: -5.0, 3.0, -7.0, 2.0
// Result: -7.0 (largest absolute value, keeping original sign)

// Step 4: Free the operation when done
MPI_Op_free(&abs_max_op);
```

**How Custom Function Works**:
```
Function Parameters Explained:

invec (IN):
├── Input vector from another process
└── Contains data being reduced

inoutvec (INOUT):
├── Input: Current accumulated result
├── Output: Updated result after combining with invec
└── You modify this in-place

len (IN):
├── Number of elements in the vectors
└── Process len elements in loop

datatype (IN):
├── MPI datatype of elements
└── Usually not needed in simple functions

Execution Pattern:
┌─────────────────────────────────────────────┐
│ Initial: inoutvec = first process data      │
│ Call 1:  combine inoutvec with process 1    │
│ Call 2:  combine inoutvec with process 2    │
│ Call 3:  combine inoutvec with process 3    │
│ Result:  inoutvec contains final result     │
└─────────────────────────────────────────────┘
```

**Important Restrictions**:
```
MUST follow these rules:

✓ Function MUST be associative
✓ If commutative, declare it (allows optimization)
✓ NO MPI calls inside reduction function
✓ NO memory allocation (keep it simple)
✓ Thread-safe (might be called from multiple threads)
✓ Modify inoutvec in-place

✗ Don't use non-associative operations
✗ Don't call MPI functions
✗ Don't depend on order if declared commutative
```

---

## **PART 9: Communicators and Groups (Slides 35-60)**

### **Introduction to Communicators (Slides 36-40)**

**What is a Communicator? (Slide 36)**

**Definition**: A communicator is a **group of processes** with associated communication context.

**Why MPI Exists**:
```
Core Purpose of MPI:
├── Make point-to-point communication portable
├── Make collective communication portable
└── Work across different machine architectures

Historical Problems MPI Solved:
├── Division of processes into logical groups
├── Avoiding message conflicts between libraries
├── Efficient communication pattern reuse
└── Safe execution guarantees
```

### **Division of Processes (Slide 37)**

**Use Case 1: Independent Coarse-Grained Tasks**
```
Example: Weather Simulation System
┌────────────────────────────────────────────┐
│ Total Processes: 90                        │
│                                            │
│ Group 1 (60 processes):                    │
│   → Predict weather patterns               │
│                                            │
│ Group 2 (30 processes):                    │
│   → Process incoming sensor data           │
└────────────────────────────────────────────┘

Each group works independently, different tasks
```

**Use Case 2: Data-Based Division**
```
Example: Matrix Operations
┌────────────────────────────────────────────┐
│ Have: Large matrix                         │
│                                            │
│ Diagonal Group:                            │
│   → Processes holding diagonal elements    │
│   → Special operations on diagonal         │
│                                            │
│ Off-Diagonal Group:                        │
│   → Rest of the matrix                     │
│   → Different operations                   │
└────────────────────────────────────────────┘

Logical naming regardless of process count
```

### **Avoiding Message Conflicts (Slide 38)**

**The Problem**:
```
Scenario: Your code uses MPI library (e.g., linear algebra library)

Your code:
├── Uses MPI_Send/Recv with tags 0-100
└── Expects specific message patterns

Linear algebra library (internally):
├── Uses MPI_Send/Recv with tags 0-50
├── You don't know library's internal tags
└── Risk: Library might consume YOUR messages!

Example Conflict:
Your code:     MPI_Recv(buffer, ..., tag=10, ...)
Library code:  MPI_Recv(buffer, ..., MPI_ANY_TAG, ...)
                       ↑
                   Oops! Library gets your message
```

**MPI Solution - Communicators**:
```
Communicators provide message isolation:

Your code uses:         MPI_COMM_WORLD
Library uses:           library_comm (separate communicator)

Result:
├── Messages in MPI_COMM_WORLD stay in MPI_COMM_WORLD
├── Messages in library_comm stay in library_comm
├── No interference between the two
└── Libraries can safely coexist

This is why communicator is required parameter!
```

### **Extensibility and Reusability (Slide 39)**

**The Problem**:
```
Computing efficient communication patterns is expensive:

Example: Broadcast on 1000 processes
├── Naive: Send to each process sequentially (O(n))
├── Tree-based: Hierarchical sends (O(log n))
└── Optimal tree depends on network topology

Computing optimal pattern:
├── Analyze network topology
├── Determine best tree structure
├── Could take significant time
└── But can be REUSED!
```

**Solution**:
```
Communicator stores precomputed patterns:

Create communicator once:
├── MPI analyzes best communication pattern
├── Stores optimal strategy in communicator
└── Takes time upfront

Use communicator many times:
├── Every Bcast uses precomputed pattern
├── No recomputation needed
├── Amortize setup cost over many operations
└── Significant performance benefit

Additional benefit:
└── Logical naming of groups (e.g., "row_comm", "col_comm")
```

### **MPI Safety (Slide 40)**

**Guarantees from Communicators**:
```
By requiring routines to be managed by communicators:

✓ Safe execution (no message conflicts)
✓ Efficient execution (optimized patterns)
✓ Portable execution (works on any MPI implementation)
✓ Predictable behavior (deterministic communication)

Without communicators:
✗ No isolation between libraries
✗ No way to optimize communication patterns
✗ Risk of message corruption
✗ Hard to reason about parallel code
```

### **Groups (Slide 41)**

**Definition**: A group is an **ordered set** of process identifiers (ranks).

**Group Characteristics**:
```
Group Properties:
├── Ordered: Processes have specific sequence
├── Contiguous ranks: Start at 0, no gaps
├── Process identifiers: Each has integer rank
└── Just a list: No communication capability yet

Group Operations:
├── Subset operations: Work on portion of processes
├── Collective operations: Can apply to group members
└── No communication: Groups alone can't communicate
```

**Special Groups**:
- `MPI_GROUP_EMPTY`: Empty group (no processes)
- `MPI_GROUP_NULL`: Returned when group is freed

**Visual Representation**:
```
Example Group (8 processes):

Group: [P0, P1, P2, P3, P4, P5, P6, P7]
Ranks:  0   1   2   3   4   5   6   7
        ↑                           ↑
    Rank 0                      Rank 7

Key points:
- Ordered from rank 0 to 7
- Contiguous (no missing ranks)
- Just identifiers (can't communicate yet)
```

### **Communicators (Slide 42)**

**Definition**: A communicator is a **handle** to an object that describes a group of processes with communication context.

**Two Types of Communicators**:

**1. Intra-communicator** (within group):
```
Intra-communicator Attributes:
├── Process group (the processes involved)
├── Topology (logical layout of processes)
│   └── Will cover next lecture
└── Used for communication WITHIN a group

Example:
┌────────────────────────────────┐
│  Intra-communicator "row_0"   │
│                                │
│  Group: [P0, P1, P2, P3]      │
│  Topology: Linear/Cartesian    │
│                                │
│  P0 ↔ P1 ↔ P2 ↔ P3            │
└────────────────────────────────┘
```

**2. Inter-communicator** (between groups):
```
Inter-communicator Attributes:
├── Pair of process groups
├── No topology
└── Used for communication BETWEEN disjoint groups

Example:
┌──────────────┐         ┌──────────────┐
│ Group A      │   ↔     │ Group B      │
│ [P0, P1, P2] │ Inter   │ [P3, P4, P5] │
└──────────────┘  comm   └──────────────┘

Processes in Group A can communicate with
processes in Group B via inter-communicator
```

**Functionality Comparison** (Slide 44):
```
┌────────────────────┬─────────────────┬──────────────────┐
│ Functionality      │ Intra-comm      │ Inter-comm       │
├────────────────────┼─────────────────┼──────────────────┤
│ Number of groups   │ 1               │ 2                │
│ Communication safe │ Yes             │ Yes              │
│ Collective ops     │ Yes             │ No               │
│ Topologies         │ Yes             │ No               │
│ Caching            │ Yes             │ Yes              │
└────────────────────┴─────────────────┴──────────────────┘
```

### **Communicators and Groups Example (Slide 43)**

**THE MOST IMPORTANT SLIDE FOR COMMUNICATORS**

```
Visual Representation of Multiple Communicators:

┌──────────────────────────────────────────────────────────┐
│                   MPI_COMM_WORLD (purple)                │
│                                                          │
│     ┌────────────────────────────────┐                  │
│     │ Comm1 (red)                    │   Comm5          │
│     │  ●P2                           │                  │
│     │      ●                          │  ●P1            │
│     │          ●P3                    │      Comm3      │
│     │                                 │      (yellow)   │
│     │         ●P0                     │      ●P0        │
│     └────────────────────────────────┘                  │
│                                                          │
│          Comm2 (blue)                                    │
│             ●                                            │
│                                                          │
│                                        Comm4 (inter)     │
└──────────────────────────────────────────────────────────┘

Process membership:
- P0: Member of MPI_COMM_WORLD, Comm3
- P1: Member of MPI_COMM_WORLD, Comm2, Comm5 (inter)
- P2: Member of MPI_COMM_WORLD, Comm1
- P3: Member of MPI_COMM_WORLD, Comm1
```

**Key Observations**:

```
4 Distinct Groups (Intracommunicators):
├── MPI_COMM_WORLD: All processes (P0, P1, P2, P3)
├── Comm1: Processes P2, P3
├── Comm2: Process P1 only
└── Comm3: Process P0 only

1 Intercommunicator:
└── Comm4/Comm5: Between two disjoint groups

Important Facts:
1. P3 is member of 2 groups (MPI_COMM_WORLD and Comm1)
   - Has rank 3 in MPI_COMM_WORLD
   - Has rank 4 in Comm1 (different rank!)

2. P2 → P1 communication options:
   - Use MPI_COMM_WORLD (intracommunicator)
   - Use Comm5 (intercommunicator)

3. P2 → P3 communication options:
   - Use MPI_COMM_WORLD (send to rank 3)
   - Use Comm1 (send to rank 4)
   
4. P0 broadcast to Comm2:
   - Use Comm4 intercommunicator
   - Reaches all processes in Comm2
```

**Critical Understanding - Different Ranks**:
```
Process P3 example:

In MPI_COMM_WORLD:
├── Processes: [P0, P1, P2, P3]
├── P3 has rank: 3
└── To send to P3: dest = 3

In Comm1:
├── Processes: [P2, P3]  (only 2 processes)
├── P3 has rank: 4 or 1 (depends on how comm was created)
└── To send to P3 in Comm1: dest = (its rank in Comm1)

Same physical process, DIFFERENT logical ranks!
```

---

## **PART 10: Group Routines (Slides 46-53)**

### **Group Management (Slide 47)**

**Group Lifecycle**:
```
Groups are initially NOT associated with communicators:
├── Can create groups independently
├── Groups alone CANNOT do message passing
└── Must create communicator from group for communication

Operations available:
├── Access groups (get size, rank)
├── Construct groups (union, intersection, etc.)
└── Destroy groups (free memory)
```



### **Group Accessors (Slides 48-49)**

**MPI_Group_size**:

```c
int MPI_Group_size(MPI_Group group, int *size);
```

- Returns number of processes in the group

**MPI_Group_rank**:

```c
int MPI_Group_rank(MPI_Group group, int *rank);
```

- Returns rank of calling process in that group

**MPI_Group_translate_ranks**:

```c
int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[],
                               MPI_Group group2, int ranks2[]);
```

- Takes array of *n* ranks in group1
- Returns corresponding ranks in group2
- Essential for converting between communicators

**Translation Example**:

```c
// Process P3 is in both MPI_COMM_WORLD and Comm1
// In MPI_COMM_WORLD, P3 has rank 3
// What is P3's rank in Comm1?

MPI_Group world_group, comm1_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);
MPI_Comm_group(Comm1, &comm1_group);

int world_ranks[1] = {3};  // P3's rank in COMM_WORLD
int comm1_ranks[1];

MPI_Group_translate_ranks(world_group, 1, world_ranks,
                          comm1_group, comm1_ranks);

// comm1_ranks[0] now contains P3's rank in Comm1
printf("Process with world rank 3 has rank %d in Comm1\n", comm1_ranks[0]);
```

**MPI_Group_compare** (Slide 49):

```c
int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
```

**Returns**:

- `MPI_IDENT`: Same processes, same rank order
- `MPI_SIMILAR`: Same processes, different rank order
- `MPI_UNEQUAL`: Different processes

**Visual Examples**:

```
MPI_IDENT:
Group1: [P0, P1, P2, P3]  ranks: 0, 1, 2, 3
Group2: [P0, P1, P2, P3]  ranks: 0, 1, 2, 3
Result: MPI_IDENT (identical)

MPI_SIMILAR:
Group1: [P0, P1, P2, P3]  ranks: 0, 1, 2, 3
Group2: [P3, P2, P1, P0]  ranks: 0, 1, 2, 3
Result: MPI_SIMILAR (same processes, different order)

MPI_UNEQUAL:
Group1: [P0, P1, P2, P3]  ranks: 0, 1, 2, 3
Group2: [P4, P5, P6, P7]  ranks: 0, 1, 2, 3
Result: MPI_UNEQUAL (different processes)
```

### **Group Constructors (Slides 50-52)**

**Overview (Slide 50)**:

```
Group construction creates new groups from existing ones:

├── Base group: Associated with MPI_COMM_WORLD
│   └── Get with: MPI_Comm_group(MPI_COMM_WORLD, &group)
│
├── Creation is LOCAL operation (no communication)
│
├── No communicator associated after creation
│   └── Must create communicator to enable communication
│
└── ALL processes must create identical group
    └── Each process runs same constructor with same parameters
```

**MPI_Comm_group**:

```c
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
```

- Returns group corresponding to communicator

**MPI_Group_union** (Slide 51):

```c
int MPI_Group_union(MPI_Group group1, MPI_Group group2, 
                    MPI_Group *newgroup);
```

- Newgroup contains all processes in group1 OR group2

**MPI_Group_intersection**:

```c
int MPI_Group_intersection(MPI_Group group1, MPI_Group group2,
                            MPI_Group *newgroup);
```

- Newgroup contains processes in BOTH group1 AND group2

**MPI_Group_difference**:

```c
int MPI_Group_difference(MPI_Group group1, MPI_Group group2,
                         MPI_Group *newgroup);
```

- Newgroup contains processes in group1 but NOT in group2

**Visual Set Operations** (Slide 52):

```
Union Example:
Group1: {0, 1, 2, 3}     Group2: {2, 3, 4, 5}

┌────────┐  ┌────────┐        ┌──────────────┐
│ 0    2 │  │ 2    4 │  Union │ 0  2  4      │
│ 1    3 │  │ 3    5 │   →    │ 1  3  5      │
└────────┘  └────────┘        └──────────────┘

Result: {0, 1, 2, 3, 4, 5}  (all unique processes)


Intersection Example:
Group1: {0, 1, 2, 3}     Group2: {2, 3, 4, 5}

┌────────┐  ┌────────┐        ┌──────┐
│ 0    2 │  │ 2    4 │  Inter │  2   │
│ 1    3 │  │ 3    5 │   →    │  3   │
└────────┘  └────────┘        └──────┘

Result: {2, 3}  (only processes in both groups)
```

### **Group Destruction (Slide 53)**

```c
int MPI_Group_free(MPI_Group *group);
```

- Frees group object
- Returns `MPI_GROUP_NULL`
- Must free groups to avoid memory leaks

---

## **PART 11: Communicator Routines (Slides 54-58)**

### **Communicator Management (Slide 55)**

**Key Principles**:

```
Communicator Access Operations:
├── LOCAL operations (no inter-process communication needed)
├── Examples: size, rank, compare
└── Fast, no network overhead

Communicator Constructors:
├── COLLECTIVE operations (all processes must participate)
├── May require inter-process communication
├── Examples: create, split, duplicate
└── All processes must call with consistent parameters

Focus: Intra-communicators
└── This lecture covers intracommunicators only
    (intercommunicators are advanced topic)
```

### **Communicator Accessors (Slide 56)**

**MPI_Comm_size**:

```c
int MPI_Comm_size(MPI_Comm comm, int *size);
```

- Returns number of processes in the communicator

**MPI_Comm_rank**:

```c
int MPI_Comm_rank(MPI_Comm comm, int *rank);
```

- Returns rank of calling process in that communicator

**MPI_Comm_compare**:

```c
int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
```

**Returns**:

- `MPI_IDENT`: comm1 and comm2 are handles for **same object**
- `MPI_CONGRUENT`: comm1 and comm2 have **same group** attribute
- `MPI_SIMILAR`: Groups have **same members, different rank order**
- `MPI_UNEQUAL`: Otherwise

**Detailed Examples**:

```c
// Example 1: MPI_IDENT
MPI_Comm comm_copy = MPI_COMM_WORLD;
MPI_Comm_compare(MPI_COMM_WORLD, comm_copy, &result);
// result == MPI_IDENT (same handle, same object)

// Example 2: MPI_CONGRUENT
MPI_Comm new_comm;
MPI_Comm_dup(MPI_COMM_WORLD, &new_comm);  // Duplicate
MPI_Comm_compare(MPI_COMM_WORLD, new_comm, &result);
// result == MPI_CONGRUENT (different handle, same group/order)

// Example 3: MPI_SIMILAR
// Assume we created new_comm with same processes but different order
MPI_Comm_compare(MPI_COMM_WORLD, reordered_comm, &result);
// result == MPI_SIMILAR (same processes, different ranks)

// Example 4: MPI_UNEQUAL  
// Assume subset_comm contains only half the processes
MPI_Comm_compare(MPI_COMM_WORLD, subset_comm, &result);
// result == MPI_UNEQUAL (different process membership)
```

### **Communicator Constructors (Slide 57)**

**MPI_Comm_dup**:

```c
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
```

- Duplicates the provided communicator
- Useful to copy then manipulate without affecting original
- Creates separate communication context

**MPI_Comm_create**:

```c
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, 
                    MPI_Comm *newcomm);
```

- Creates new intracommunicator using **subset** of comm
- Only processes in group get valid newcomm
- Processes not in group get MPI_COMM_NULL

**MPI_Comm_split** (Most Important):

```c
int MPI_Comm_split(MPI_Comm comm, int colour, int key, 
                   MPI_Comm *newcomm);
```

**Parameters**:

- `comm` (IN): Communicator to split
- `colour` (IN): Processes with same colour go in same new communicator
- `key` (IN): Determines ranking within new communicator (lower key = lower rank)
- `newcomm` (OUT): New communicator

**How MPI_Comm_split Works**:

```
Splitting Process:

Step 1: Each process provides a colour
├── Processes with SAME colour → same new communicator
├── Processes with DIFFERENT colour → different communicators
└── Special: colour = MPI_UNDEFINED → process gets MPI_COMM_NULL

Step 2: Within each colour group, rank by key
├── Lower key value → lower rank in new communicator
├── Same key → rank by original rank in comm
└── key only matters within same colour group

Result: Multiple new communicators created simultaneously
```

**Visual Example**:

```
Original: MPI_COMM_WORLD with 16 processes (ranks 0-15)

MPI_Comm_split(MPI_COMM_WORLD, rank/4, rank, &new_comm);
                                ↑       ↑
                            colour    key

Colour assignment:
├── Ranks 0-3:   colour = 0
├── Ranks 4-7:   colour = 1
├── Ranks 8-11:  colour = 2
└── Ranks 12-15: colour = 3

Result: 4 new communicators

New Comm 0: [P0, P1, P2, P3]    (colour 0, ranks 0-3)
New Comm 1: [P4, P5, P6, P7]    (colour 1, ranks 0-3)
New Comm 2: [P8, P9, P10, P11]  (colour 2, ranks 0-3)
New Comm 3: [P12, P13, P14, P15] (colour 3, ranks 0-3)

Note: Ranks in new communicators reset to 0-3
```

### **Communicator Split Example (Slide 58)**

**Complete Working Code**:

```c
#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int myid, numprocs;
    int color, broad_val, new_id, new_nodes;
    MPI_Comm New_Comm;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    // Split processes into odd and even groups
    color = myid % 2;  // 0 for even ranks, 1 for odd ranks
    
    MPI_Comm_split(MPI_COMM_WORLD, color, myid, &New_Comm);
    MPI_Comm_rank(New_Comm, &new_id);
    MPI_Comm_size(New_Comm, &new_nodes);
    
    // Broadcast within new communicator
    if(new_id == 0) 
        broad_val = color;
    
    MPI_Bcast(&broad_val, 1, MPI_INT, 0, New_Comm);
    
    printf("old_proc %d has new rank %d, broad_val:%d\n", 
           myid, new_id, broad_val);
    
    MPI_Finalize();
    return 0;
}
```

**Step-by-Step Execution** (8 processes):

```
STEP 1: Original Configuration
┌──────────────────────────────────────────────────────────┐
│ MPI_COMM_WORLD: [P0, P1, P2, P3, P4, P5, P6, P7]       │
│ Ranks:           0   1   2   3   4   5   6   7         │
└──────────────────────────────────────────────────────────┘

STEP 2: Calculate Colour (color = myid % 2)
┌──────────────────────────────────────────────────────────┐
│ P0: color = 0 % 2 = 0  (EVEN)                           │
│ P1: color = 1 % 2 = 1  (ODD)                            │
│ P2: color = 2 % 2 = 0  (EVEN)                           │
│ P3: color = 3 % 2 = 1  (ODD)                            │
│ P4: color = 4 % 2 = 0  (EVEN)                           │
│ P5: color = 5 % 2 = 1  (ODD)                            │
│ P6: color = 6 % 2 = 0  (EVEN)                           │
│ P7: color = 7 % 2 = 1  (ODD)                            │
└──────────────────────────────────────────────────────────┘

STEP 3: MPI_Comm_split Creates Two New Communicators
┌──────────────────────────────────────────────────────────┐
│ New_Comm (color=0, EVEN processes):                     │
│   Members: [P0, P2, P4, P6]                             │
│   New ranks: 0   1   2   3                              │
│                                                          │
│ New_Comm (color=1, ODD processes):                      │
│   Members: [P1, P3, P5, P7]                             │
│   New ranks: 0   1   2   3                              │
└──────────────────────────────────────────────────────────┘

Key observation: SAME variable name "New_Comm" but 
                 DIFFERENT communicators for even/odd groups!

STEP 4: Rank Mapping
┌────────────────────────────────────────────────────────┐
│ Original → New Rank Mapping:                           │
│                                                        │
│ EVEN group (color=0):                                  │
│   P0: old rank 0 → new rank 0                         │
│   P2: old rank 2 → new rank 1                         │
│   P4: old rank 4 → new rank 2                         │
│   P6: old rank 6 → new rank 3                         │
│                                                        │
│ ODD group (color=1):                                   │
│   P1: old rank 1 → new rank 0                         │
│   P3: old rank 3 → new rank 1                         │
│   P5: old rank 5 → new rank 2                         │
│   P7: old rank 7 → new rank 3                         │
└────────────────────────────────────────────────────────┘

STEP 5: Broadcast Within Each New Communicator
┌──────────────────────────────────────────────────────────┐
│ EVEN group (New_Comm for color=0):                      │
│   Rank 0 (P0): broad_val = 0 (color)                    │
│   MPI_Bcast sends 0 to all in EVEN group                │
│   Result: All EVEN processes have broad_val = 0         │
│                                                          │
│ ODD group (New_Comm for color=1):                       │
│   Rank 0 (P1): broad_val = 1 (color)                    │
│   MPI_Bcast sends 1 to all in ODD group                 │
│   Result: All ODD processes have broad_val = 1          │
└──────────────────────────────────────────────────────────┘

FINAL OUTPUT (8 processes):
old_proc 0 has new rank 0, broad_val:0
old_proc 2 has new rank 1, broad_val:0
old_proc 4 has new rank 2, broad_val:0
old_proc 6 has new rank 3, broad_val:0
old_proc 1 has new rank 0, broad_val:1
old_proc 3 has new rank 1, broad_val:1
old_proc 5 has new rank 2, broad_val:1
old_proc 7 has new rank 3, broad_val:1
```

**Critical Understanding**:

```
Why This Works:

1. MPI_Comm_split creates TWO separate communicators
   ├── Each group thinks it has ranks 0, 1, 2, 3
   └── But they're ISOLATED from each other

2. Broadcast happens INDEPENDENTLY in each group
   ├── EVEN group: broadcast value 0
   ├── ODD group: broadcast value 1
   └── No interference between groups

3. Key parameter: color = myid % 2
   ├── Determines which group process joins
   ├── Processes with same color → same group
   └── Simple way to create even/odd division

4. Key parameter: key = myid
   ├── Determines rank order within new group
   ├── Lower myid → lower new rank
   └── Preserves relative ordering
```

**Common Use Cases for MPI_Comm_split**:

```
Row/Column Decomposition (2D arrays):
├── Split by row: color = row_index
├── Split by column: color = col_index
└── Enables efficient row-wise and column-wise operations

Hierarchical Processing:
├── Split by level: color = hierarchy_level
└── Different groups handle different granularity

Algorithm Phases:
├── Split by task type: color = task_id
└── Different groups execute different algorithm phases

Load Balancing:
├── Split by workload: color = workload_category
└── Isolate processes with similar characteristics
```

---

## **PART 12: Summary (Slide 60)**

### **Key Takeaways - Collective Communication**

**Collective Communication Benefits**:

```
✓ Simplifies many common patterns
✓ More efficient than multiple point-to-point calls
✓ Broadcast/Reduce, Scatter/Gather eliminate complex code
✓ Implicit synchronization (barrier-like behavior)
✓ MPI-optimized implementations (tree-based, etc.)
```

**Communication Patterns Summary**:

```
One → All:
├── MPI_Bcast: Same data to all
└── MPI_Scatter: Different data to each

All → One:
├── MPI_Gather: Collect from all
└── MPI_Reduce: Collect and operate

All → All:
├── MPI_Allgather: Everyone gets everything
├── MPI_Allreduce: Everyone gets reduced result
└── MPI_Alltoall: Complete exchange (transpose)

Specialized:
├── MPI_Scan: Prefix operation
└── MPI_Reduce_scatter: Reduce then scatter
```

### **Key Takeaways - Communicators**

**Communicators and Groups**:

```
✓ Communicators organize processes into logical groups
✓ Enable message isolation (no conflicts between libraries)
✓ Allow efficient pattern reuse (precomputed topologies)
✓ Communicators = Groups + Communication Context
✓ Groups alone cannot communicate (need communicator)
```

**Important Concepts**:

```
Groups:
├── Ordered set of process identifiers
├── Local operations (no communication)
├── Can be combined: union, intersection, difference
└── Must create communicator for message passing

Communicators:
├── Intra-communicator: Within single group
├── Inter-communicator: Between two groups
├── Provide communication context
└── Essential parameter for all MPI communication

Process Ranks:
├── Same process can have DIFFERENT ranks in different communicators
├── Must translate ranks when moving between communicators
└── Use MPI_Group_translate_ranks for conversion
```

**Practical Applications**:

```
When to use custom communicators:

✓ Divide processes by task type
✓ Organize processes by data partition (rows, columns)
✓ Isolate library communication from application
✓ Create hierarchical process structures
✓ Implement multi-phase algorithms
✓ Enable efficient subset operations
```

---

## **CRITICAL EXAM REMINDERS**

### **Most Important Slides for Exam**

**Slide 14 - Global Communication Overview**:

```
MEMORIZE THIS SLIDE:
├── Visual patterns for Broadcast, Scatter, Gather
├── Allgather, Alltoall data movement
├── Understanding data flow direction
└── This is THE reference for all collective operations
```

**Slide 43 - Communicators and Groups**:

```
MEMORIZE THIS SLIDE:
├── Multiple communicators coexisting
├── Process membership in multiple groups
├── Different ranks in different communicators
└── Inter-communicator vs intra-communicator
```

### **Common Exam Mistakes to Avoid**

**Collective Communication Mistakes**:

```
✗ Forgetting ALL processes must call collective operations
✗ Confusing which parameters matter at root vs non-root
✗ Not allocating receive buffers on non-root processes
✗ Using different counts/types on different processes
✗ Assuming specific execution order within collective ops
✗ Forgetting implicit barrier behavior
```

**Communicator Mistakes**:

```
✗ Thinking same process has same rank in all communicators
✗ Not understanding MPI_Comm_split color/key parameters
✗ Forgetting groups alone cannot communicate
✗ Not freeing communicators/groups (memory leaks)
✗ Calling collective on wrong communicator
```

### **Function Parameter Patterns**

**Standard Pattern for Most Collective Ops**:

```c
MPI_Operation(
    sendbuf,      // IN: What you're sending
    sendcount,    // IN: How many elements
    sendtype,     // IN: Type of elements
    recvbuf,      // OUT: Where result goes
    recvcount,    // IN: How many to receive
    recvtype,     // IN: Type of received elements
    root,         // IN: Root process rank (if applicable)
    comm          // IN: Communicator
);
```

**Remember**:

- "Send" parameters often only significant at root
- "Recv" parameters often only significant at root (except All* operations)
- Count is **per process**, not total
- Must match datatypes between send and receive

### **Performance Considerations**

**When Collective Operations Shine**:

```
✓ Large process counts (tree-based implementations)
✓ Regular communication patterns
✓ All processes participating
✓ Repeated operations (amortize setup cost)
```

**When to Consider Point-to-Point Instead**:

```
✓ Irregular communication patterns
✓ Only few processes communicating
✓ Non-uniform data sizes
✓ Dynamic/unpredictable communication
```

---

## **PRACTICAL CODING TIPS**

### **Debugging Collective Operations**

**Common Issues**:

```
1. Deadlock from missing collective call
   └── Solution: Ensure ALL processes call collective

2. Wrong data received
   └── Solution: Check root parameter, verify ranks

3. Segmentation fault
   └── Solution: Allocate buffers correctly on all processes

4. Unexpected values
   └── Solution: Verify send/recv counts match expectations
```

**Debugging Strategy**:

```c
// Print from each process to verify participation
printf("Rank %d: Before MPI_Bcast\n", rank);
fflush(stdout);

MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

printf("Rank %d: After MPI_Bcast, value=%d\n", rank, value);
fflush(stdout);
```

### **Best Practices**

**Code Organization**:

```c
// Good: Clear separation by rank
if (rank == ROOT) {
    // Root-specific initialization
    // Allocate send buffers
} else {
    // Non-root initialization  
    // May not need send buffers
}

// Collective operation (ALL processes call)
MPI_Scatter(sendbuf, sendcount, MPI_INT,
            recvbuf, recvcount, MPI_INT,
            ROOT, MPI_COMM_WORLD);

// All processes now have data in recvbuf
```

**Error Checking**:

```c
int ierr = MPI_Bcast(&data, count, MPI_INT, 0, MPI_COMM_WORLD);
if (ierr != MPI_SUCCESS) {
    fprintf(stderr, "Error in MPI_Bcast: %d\n", ierr);
    MPI_Abort(MPI_COMM_WORLD, ierr);
}
```

**Memory Management**:

```c
// Always free communicators and groups
MPI_Comm_free(&custom_comm);
MPI_Group_free(&custom_group);

// Free custom operations
MPI_Op_free(&custom_op);
```

---

## **ASSIGNMENT 2 CONNECTION**

### **Relevant to 2D Convolution with MPI**

**Expected Usage**:

```
Data Distribution:
├── MPI_Scatter: Distribute image portions to processes
└── Each process works on its assigned image region

Result Collection:
├── MPI_Gather: Collect processed portions
└── Root assembles final result

Border Exchange:
├── MPI_Send/Recv: Exchange border pixels (halo exchange)
└── Necessary for convolution at boundaries

Process Organization:
├── Could use MPI_Comm_split for row/column groups
└── Enables efficient 2D decomposition

Performance Measurement:
├── MPI_Barrier: Synchronize before timing
├── MPI_Reduce: Collect timing statistics
└── Analyze scaling performance
```

**Key Patterns for Assignment**:

```c
// 1. Distribute work
MPI_Scatter(full_image, portion_size, MPI_FLOAT,
            my_portion, portion_size, MPI_FLOAT,
            0, MPI_COMM_WORLD);

// 2. Exchange borders with neighbors
MPI_Sendrecv(top_border, border_size, MPI_FLOAT, top_neighbor, TAG,
             bottom_border, border_size, MPI_FLOAT, bottom_neighbor, TAG,
             MPI_COMM_WORLD, &status);

// 3. Collect results
MPI_Gather(my_result, portion_size, MPI_FLOAT,
           full_result, portion_size, MPI_FLOAT,
           0, MPI_COMM_WORLD);
```

---

## **QUICK REFERENCE TABLES**

### **Collective Communication Quick Reference**

```
┌──────────────┬─────────────┬──────────┬─────────────────────┐
│ Function     │ Pattern     │ Root?    │ Result Location     │
├──────────────┼─────────────┼──────────┼─────────────────────┤
│ Bcast        │ 1 → All     │ Yes      │ All processes       │
│ Scatter      │ 1 → All     │ Yes      │ Each gets portion   │
│ Gather       │ All → 1     │ Yes      │ Root only           │
│ Allgather    │ All → All   │ No       │ All processes       │
│ Reduce       │ All → 1     │ Yes      │ Root only           │
│ Allreduce    │ All → All   │ No       │ All processes       │
│ Scan         │ Prefix      │ No       │ All (cumulative)    │
│ Alltoall     │ All ↔ All   │ No       │ All (transposed)    │
└──────────────┴─────────────┴──────────┴─────────────────────┘
```

### **Reduction Operations Quick Reference**

```
┌──────────────┬───────────────────────────────────────────┐
│ Operation    │ Description                               │
├──────────────┼───────────────────────────────────────────┤
│ MPI_MAX      │ Maximum value                             │
│ MPI_MIN      │ Minimum value                             │
│ MPI_SUM      │ Sum of values                             │
│ MPI_PROD     │ Product of values                         │
│ MPI_LAND     │ Logical AND                               │
│ MPI_LOR      │ Logical OR                                │
│ MPI_BAND     │ Bitwise AND                               │
│ MPI_BOR      │ Bitwise OR                                │
│ MPI_MAXLOC   │ Max value + owner rank                    │
│ MPI_MINLOC   │ Min value + owner rank                    │
└──────────────┴───────────────────────────────────────────┘
```

### **Group/Communicator Operations Quick Reference**

```
┌────────────────────┬─────────────────────────────────────┐
│ Function           │ Purpose                             │
├────────────────────┼─────────────────────────────────────┤
│ MPI_Comm_group     │ Get group from communicator         │
│ MPI_Comm_size      │ Get number of processes             │
│ MPI_Comm_rank      │ Get my rank in communicator         │
│ MPI_Comm_split     │ Split into multiple communicators   │
│ MPI_Comm_create    │ Create from subset                  │
│ MPI_Comm_dup       │ Duplicate communicator              │
│ MPI_Group_union    │ Combine two groups                  │
│ MPI_Group_inter    │ Intersection of groups              │
│ MPI_Group_diff     │ Difference of groups                │
└────────────────────┴─────────────────────────────────────┘
```

---

**End of Lecture 9 Notes**

*These comprehensive notes cover all slides from Lecture 9 on MPI Collective Communication and Communicators. Study the visual diagrams carefully, understand the data flow patterns, and practice writing code using these collective operations. The concepts of communicators and groups are fundamental to advanced MPI programming.*# CITS3402 Lecture 9: MPI Collective Communication and Communicators