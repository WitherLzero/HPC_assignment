#!/bin/bash
#SBATCH --job-name=kaya_info
#SBATCH --partition=cits3402
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1GB
#SBATCH --output=kaya.info
#SBATCH --error=kaya.err

echo "===== HOSTNAME ====="
hostname

echo "===== CPU INFO ====="
lscpu

echo "===== MEMORY INFO ====="
free -h

echo "===== OS INFO ====="
cat /etc/os-release

echo "===== KERNEL INFO ====="
uname -a

echo "===== COMPILER INFO ====="
gcc --version

echo "===== CPU CACHE INFO ====="
lscpu | grep cache

echo "===== DETAILED MEMORY INFO ====="
cat /proc/meminfo | head -20

echo "===== CPU MODEL INFO ====="
cat /proc/cpuinfo | grep "model name" | head -1

echo "===== NVIDIA GPU INFO (if available) ====="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No NVIDIA GPU found or nvidia-smi not available"
fi
