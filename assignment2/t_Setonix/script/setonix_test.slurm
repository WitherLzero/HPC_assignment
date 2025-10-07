#!/bin/bash
#SBATCH --job-name=conv2d_setonix
#SBATCH --account=courses0101
#SBATCH --partition=work

# ========================================================================
# RESOURCE ALLOCATION - Choose ONE configuration
# ========================================================================

#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16

#SBATCH --time=00:15:00
#SBATCH --output=setonix_results/setonix_hpc_%j.txt
#SBATCH --error=setonix_results/setonix_hpc_error_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=24745401@student.uwa.edu.au

# CITS3402 Assignment 2 - Setonix Performance Testing (Hybrid MPI+OpenMP)
# ========================================================================
# Hardware: AMD EPYC 7763 (2 sockets × 64 cores × 2 threads = 256 logical cores/node)
# 
#
# Note: Using ALL logical cores (including hyperthreading)
# Tests below auto-adapt to ntasks allocation

echo "============================================"
echo "Setonix HPC Performance Testing - Assignment 2"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES × AMD EPYC 7763"
echo "Node list: $SLURM_NODELIST"
echo "MPI tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "Start time: $(date)"
echo "============================================"
echo ""

# Load Cray programming environment
module load PrgEnv-gnu
module load cray-mpich/8.1.32

echo "Loaded modules:"
module list 2>&1
echo ""

# Temporarily update Makefile to use Cray compiler wrapper
echo "Updating Makefile for Cray environment..."
cp Makefile Makefile.backup
sed -i 's/CC = mpicc/CC = cc/' Makefile

echo "Building project..."
make clean && make
BUILD_STATUS=$?

# Restore original Makefile
mv Makefile.backup Makefile

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi
echo "Build completed successfully."
echo ""

mkdir -p results

# Calculate core counts for scaling tests
TOTAL_CORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
HALF_CORES=$((TOTAL_CORES / 2))
QUARTER_CORES=$((TOTAL_CORES / 4))
HALF_TASKS=$((SLURM_NTASKS / 2))
QUARTER_TASKS=$((SLURM_NTASKS / 4))

# ==========================================
# SECTION 1: Strong Scaling (Fixed 50000×50000)
# ==========================================
echo "============================================"
echo "SECTION 1: STRONG SCALING"
echo "Fixed problem: 50000×50000, kernel 3×3"
echo "============================================"
echo ""

H=50000
W=50000
KH=3
KW=3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[Strong 1.1] Hybrid - $SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
echo "[Strong 1.2] Hybrid - $SLURM_NTASKS proc × $OMP_NUM_THREADS threads = $HALF_CORES cores"
srun -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
echo "[Strong 1.3] Hybrid - $SLURM_NTASKS proc × $OMP_NUM_THREADS threads = $QUARTER_CORES cores"
srun -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[Strong 1.4] Hybrid - $HALF_TASKS proc × $SLURM_CPUS_PER_TASK threads = $HALF_CORES cores"
srun -n $HALF_TASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
echo "[Strong 1.5] Hybrid - $HALF_TASKS proc × $OMP_NUM_THREADS threads = $QUARTER_CORES cores"
srun -n $HALF_TASKS -c $OMP_NUM_THREADS ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
echo "[Strong 1.6] Hybrid - $HALF_TASKS proc × $OMP_NUM_THREADS threads = 32 cores"
srun -n $QUARTER_TASKS -c $OMP_NUM_THREADS ./build/conv_stride_test -H $H -W $W -kH $KH -kW $KW -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 2: Weak Scaling
# ==========================================
echo "============================================"
echo "SECTION 2: WEAK SCALING"
echo "Proportional matrix size × core count"
echo "============================================"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[Weak 2.1] $QUARTER_CORES cores: 50000×50000 matrix ($QUARTER_TASKS proc × $SLURM_CPUS_PER_TASK threads)"
srun -n $QUARTER_TASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H 50000 -W 50000 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Weak 2.2] $HALF_CORES cores: 70711×70711 matrix ($HALF_TASKS proc × $SLURM_CPUS_PER_TASK threads)"
srun -n $HALF_TASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H 70711 -W 70711 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Weak 2.3] $TOTAL_CORES cores: 100000×100000 matrix ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H 100000 -W 100000 -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 3: Stride Impact (Large Matrix)
# ==========================================
echo "============================================"
echo "SECTION 3: STRIDE IMPACT"
echo "Matrix: 80000×80000, kernel 3×3"
echo "============================================"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
H_STRIDE=80000
W_STRIDE=80000

echo "[Stride 3.1] Stride 1×1 ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Stride 3.2] Stride 2×2 ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 2 -sW 2 -m -t
echo ""

echo "[Stride 3.3] Stride 3×3 ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 3 -sW 3 -m -t
echo ""

# ==========================================
# SECTION 4: Kernel Size Impact (Large Matrix)
# ==========================================
echo "============================================"
echo "SECTION 4: KERNEL SIZE IMPACT"
echo "Matrix: 80000×80000, stride 1×1"
echo "============================================"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[Kernel 4.1] 3×3 kernel ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Kernel 4.2] 5×5 kernel ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 5 -kW 5 -sH 1 -sW 1 -m -t
echo ""

echo "[Kernel 4.3] 7×7 kernel ($SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores)"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_STRIDE -W $W_STRIDE -kH 7 -kW 7 -sH 1 -sW 1 -m -t
echo ""

# ==========================================
# SECTION 5: Multi-Node Scaling
# ==========================================
echo "============================================"
echo "SECTION 5: MULTI-NODE SCALING"
echo "Matrix: 100000×100000, kernel 3×3"
echo "============================================"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
H_NODE=100000
W_NODE=100000

SINGLE_NODE_TASKS=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
SINGLE_NODE_CORES=$((SINGLE_NODE_TASKS * SLURM_CPUS_PER_TASK))

echo "[Node 5.1] Single node: $SINGLE_NODE_TASKS proc × $SLURM_CPUS_PER_TASK threads = $SINGLE_NODE_CORES cores"
srun -n $SINGLE_NODE_TASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_NODE -W $W_NODE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "[Node 5.2] $SLURM_JOB_NUM_NODES nodes: $SLURM_NTASKS proc × $SLURM_CPUS_PER_TASK threads = $TOTAL_CORES cores"
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ./build/conv_stride_test -H $H_NODE -W $W_NODE -kH 3 -kW 3 -sH 1 -sW 1 -m -t
echo ""

echo "============================================"
echo "SETONIX PERFORMANCE TESTING COMPLETED"
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "============================================"
