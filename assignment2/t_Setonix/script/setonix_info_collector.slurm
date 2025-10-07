#!/bin/bash
#SBATCH --job-name=setonix_info
#SBATCH --account=pawsey0001
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=setonix_hardware_info.txt
#SBATCH --error=setonix_hardware_info_err.txt

# Setonix Hardware Information Collection Script
# ==============================================
# This script collects detailed hardware information from compute nodes
# to help determine optimal SLURM parameters for performance testing

echo "============================================"
echo "SETONIX HARDWARE INFORMATION COLLECTOR"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Collection time: $(date)"
echo "Node: $(hostname)"
echo ""

echo "============================================"
echo "1. SLURM ENVIRONMENT VARIABLES"
echo "============================================"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo ""

echo "============================================"
echo "2. CPU INFORMATION (from lscpu)"
echo "============================================"
lscpu
echo ""

echo "============================================"
echo "3. CPU DETAILS (parsed)"
echo "============================================"
echo "Architecture: $(lscpu | grep 'Architecture:' | awk '{print $2}')"
echo "CPU(s): $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
echo "Thread(s) per core: $(lscpu | grep 'Thread' | awk '{print $4}')"
echo "Core(s) per socket: $(lscpu | grep 'Core(s)' | awk '{print $4}')"
echo "Socket(s): $(lscpu | grep 'Socket' | awk '{print $2}')"
echo "NUMA node(s): $(lscpu | grep 'NUMA node(s)' | awk '{print $3}')"
echo "CPU max MHz: $(lscpu | grep 'CPU max MHz' | awk '{print $4}')"
echo "CPU model: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo ""

echo "============================================"
echo "4. MEMORY INFORMATION"
echo "============================================"
echo "Total memory (from /proc/meminfo):"
cat /proc/meminfo | grep MemTotal
echo ""
echo "Available memory:"
cat /proc/meminfo | grep MemAvailable
echo ""
echo "Memory info (free -h):"
free -h
echo ""

echo "============================================"
echo "5. PARTITION INFORMATION (sinfo)"
echo "============================================"
sinfo -o "%P %D %c %m %l %N"
echo ""

echo "============================================"
echo "6. DETAILED PARTITION INFO (work partition)"
echo "============================================"
scontrol show partition work
echo ""

echo "============================================"
echo "7. NODE DETAILED INFORMATION"
echo "============================================"
scontrol show node $(hostname) 2>/dev/null || echo "Cannot get detailed node info on compute node"
echo ""

echo "============================================"
echo "8. LOADED MODULES"
echo "============================================"
module list
echo ""

echo "============================================"
echo "9. AVAILABLE GCC VERSIONS"
echo "============================================"
module spider gcc 2>&1 | head -30
echo ""

echo "============================================"
echo "10. AVAILABLE MPI VERSIONS"
echo "============================================"
echo "OpenMPI:"
module spider openmpi 2>&1 | head -20
echo ""
echo "Cray-MPICH:"
module spider cray-mpich 2>&1 | head -20
echo ""

echo "============================================"
echo "11. ACCOUNT INFORMATION"
echo "============================================"
echo "User: $USER"
sacctmgr show user $USER -s
echo ""

echo "============================================"
echo "12. CPU CALCULATION"
echo "============================================"
TOTAL_CPUS=$(lscpu | grep '^CPU(s):' | awk '{print $2}')
THREADS_PER_CORE=$(lscpu | grep 'Thread' | awk '{print $4}')
CORES_PER_SOCKET=$(lscpu | grep 'Core(s)' | awk '{print $4}')
SOCKETS=$(lscpu | grep 'Socket' | awk '{print $2}')

PHYSICAL_CORES=$((CORES_PER_SOCKET * SOCKETS))
TOTAL_MEMORY_KB=$(cat /proc/meminfo | grep MemTotal | awk '{print $2}')
TOTAL_MEMORY_GB=$((TOTAL_MEMORY_KB / 1024 / 1024))

echo "Total logical CPUs: $TOTAL_CPUS"
echo "Physical cores: $PHYSICAL_CORES"
echo "Sockets: $SOCKETS"
echo "Cores per socket: $CORES_PER_SOCKET"
echo "Threads per core: $THREADS_PER_CORE"
echo "Total memory: ${TOTAL_MEMORY_GB} GB"
echo ""

echo "============================================"
echo "13. RECOMMENDED SLURM CONFIGURATIONS"
echo "============================================"
echo "For 1 node (all cores):"
echo "  --nodes=1"
echo "  --ntasks=8 --cpus-per-task=32  (8 MPI × 32 threads = 256 cores)"
echo "  --ntasks=16 --cpus-per-task=16 (16 MPI × 16 threads = 256 cores)"
echo "  --ntasks=32 --cpus-per-task=8  (32 MPI × 8 threads = 256 cores)"
echo ""
echo "For 2 nodes (512 total cores):"
echo "  --nodes=2 --ntasks-per-node=8"
echo "  --ntasks=16 --cpus-per-task=32  (16 MPI × 32 threads = 512 cores)"
echo "  --ntasks=16 --cpus-per-task=16  (16 MPI × 16 threads = 256 cores)"
echo ""
echo "For 4 nodes (1024 total cores):"
echo "  --nodes=4 --ntasks-per-node=8"
echo "  --ntasks=32 --cpus-per-task=32  (32 MPI × 32 threads = 1024 cores)"
echo ""

echo "Memory per CPU (if needed):"
echo "  Total node memory: ${TOTAL_MEMORY_GB} GB"
echo "  Safety factor 0.9: $((TOTAL_MEMORY_GB * 9 / 10)) GB usable"
echo "  For 8 tasks/node: ~$((TOTAL_MEMORY_GB * 9 / 10 / 8)) GB per task"
echo "  For 16 tasks/node: ~$((TOTAL_MEMORY_GB * 9 / 10 / 16)) GB per task"
echo ""

echo "============================================"
echo "INFORMATION COLLECTION COMPLETE"
echo "============================================"
echo "End time: $(date)"
echo ""
echo "NEXT STEPS:"
echo "1. Read this file: cat setonix_hardware_info.txt"
echo "2. Send the output to the AI assistant"
echo "3. AI will create optimized SLURM scripts based on this info"
echo "============================================"
