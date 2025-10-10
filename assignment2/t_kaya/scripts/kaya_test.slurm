#!/bin/bash
#SBATCH --job-name=conv2d_hpc
#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=cits3402
#SBATCH --output=t_kaya/results/kaya_hpc_%j.txt
#SBATCH --error=t_kaya/results/kaya_hpc_error_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24745401@student.uwa.edu.au

# CITS3402 Assignment 2 - HPC Performance Testing (Hybrid MPI+OpenMP)
# ====================================================================
# Focus: Large-scale matrix performance with hybrid parallelization
# System: 2 nodes × 48 cores/node = 96 cores total
# Resource allocation: 24 MPI tasks × 8 CPUs/task = 192 cores
# Note: Testing from single-node (48 cores) up to full 2-node (192 cores)

echo "============================================"
echo "HPC Performance Testing - Assignment 2"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES (2 nodes × 48 cores/node)"
echo "Available: 192 cores total (24 tasks × 8 threads/task)"
echo "Start time: $(date)"
echo "============================================"
echo ""

module load gcc/14.3
module load openmpi/5.0.5

echo "Building project..."
make clean && make
echo "Build completed."
echo ""

mkdir -p results

# ==========================================
# SECTION 1: Strong Scaling (Fixed 50000×50000)
# ==========================================
echo "============================================"
echo "SECTION 1: STRONG SCALING"
echo "Fixed problem: 50000×50000, kernel 3×3"
echo "Scaling from 48 cores (1 node) to 192 cores (2 nodes)"
echo "============================================"
echo ""

H=50000
W=50000
KH=3
KW=3

# Test different core counts with optimal thread counts
export OMP_NUM_THREADS=8
echo "[Strong 1.1] 48 cores: 6 proc × 8 threads (single node)"
mpirun -np 6 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

echo "[Strong 1.2] 96 cores: 12 proc × 8 threads (2 nodes)"
mpirun -np 12 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

echo "[Strong 1.3] 144 cores: 18 proc × 8 threads (2 nodes)"
mpirun -np 18 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

echo "[Strong 1.4] 192 cores: 24 proc × 8 threads (2 nodes)"
mpirun -np 24 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

# Additional test with different thread counts
export OMP_NUM_THREADS=4
echo "[Strong 1.5] 96 cores: 24 proc × 4 threads (2 nodes)"
mpirun -np 24 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=2
echo "[Strong 1.6] 48 cores: 24 proc × 2 threads (2 nodes)"
mpirun -np 24 ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KH -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 2: Weak Scaling
# ==========================================
echo "============================================"
echo "SECTION 2: WEAK SCALING"
echo "Proportional matrix size × core count"
echo "Base: ~5e9 ops per core (50K×50K / 48 cores)"
echo "============================================"
echo ""

export OMP_NUM_THREADS=8

echo "[Weak 2.1] 48 cores: 50000×50000 matrix (6 proc × 8 threads)"
mpirun -np 6 ./build/conv_stride_test -H 50000 -W 50000 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Weak 2.2] 96 cores: 70711×70711 matrix (12 proc × 8 threads)"
mpirun -np 12 ./build/conv_stride_test -H 70711 -W 70711 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Weak 2.3] 144 cores: 86603×86603 matrix (18 proc × 8 threads)"
mpirun -np 18 ./build/conv_stride_test -H 86603 -W 86603 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Weak 2.4] 192 cores: 100000×100000 matrix (24 proc × 8 threads)"
mpirun -np 24 ./build/conv_stride_test -H 100000 -W 100000 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 3: Stride Impact (Large Matrix)
# ==========================================
echo "============================================"
echo "SECTION 3: STRIDE IMPACT"
echo "Matrix: 80000×80000, kernel 3×3"
echo "Full 192 cores (24 proc × 8 threads)"
echo "============================================"
echo ""

export OMP_NUM_THREADS=8
H_STRIDE=80000
W_STRIDE=80000

echo "[Stride 3.1] Stride 1×1 (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Stride 3.2] Stride 2×2 (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 2 -sW 2 -m -t
echo ""

echo "[Stride 3.3] Stride 3×3 (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 3 -sW 3 -m -t
echo ""

# ==========================================
# SECTION 4: Kernel Size Impact (Large Matrix)
# ==========================================
echo "============================================"
echo "SECTION 4: KERNEL SIZE IMPACT"
echo "Matrix: 80000×80000, stride 1×1"
echo "Full 192 cores (24 proc × 8 threads)"
echo "============================================"
echo ""

export OMP_NUM_THREADS=8

echo "[Kernel 4.1] 3×3 kernel (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Kernel 4.2] 5×5 kernel (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 5 -kW 5 -sH 1 -sW 1 -m -t
echo ""

echo "[Kernel 4.3] 7×7 kernel (24 proc × 8 threads = 192 cores)"
mpirun -np 24 ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 7 -kW 7 -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 5: Multi-Node Scaling
# ==========================================
echo "============================================"
echo "SECTION 5: MULTI-NODE SCALING"
echo "Matrix: 100000×100000, kernel 3×3"
echo "Compare single-node vs 2-node performance"
echo "============================================"
echo ""

export OMP_NUM_THREADS=8
H_NODE=100000
W_NODE=100000

echo "[Node 5.1] Single node, 96 cores: 12 proc × 8 threads"
mpirun -np 12 ./build/conv_stride_test -H $H_NODE -W $H_NODE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Node 5.2] 2 nodes, 192 cores: 24 proc × 8 threads"
mpirun -np 24 ./build/conv_stride_test -H $H_NODE -W $H_NODE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "============================================"
echo "HPC PERFORMANCE TESTING COMPLETED"
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "============================================"
