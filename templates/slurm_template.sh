#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=gene_scan_%j.out
#SBATCH --error=gene_scan_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --account={account}

module purge
module load intel-oneapi-compilers-classic/2021.10.0
module load intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1
module load hdf5/1.14.3--intel-oneapi-mpi--2021.12.1--oneapi--2024.1.0
module load git cmake

# $1 = run directory created by the scan builder, containing the "parameters" file
RUN_DIR=$1

if [ -z "$RUN_DIR" ]; then
    echo "Error: no run directory passed to run_gene_scan.sh"
    exit 1
fi

cd "$RUN_DIR" || exit 1
srun --mpi=pmi2 {gene_executable}