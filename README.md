# Install Petsc
Assume intel-oneapi-mpi and mkl are proper, which means that you can correctly find mpiicc(mpiicx), mpiifort(mpiifx), mpiicpc(mpiicpx), and ${MKLROOT}.
The project may need slepc, hdf5, and one mpi direct solver (superLU_dist, mumps, cpardiso).
Try the follow command to build petsc, and you may specify correct optimization flags regarding the CPU architecture.

./configure \
PETSC_ARCH=linux-oneapi-complex-opt \
--CC=mpiicx \
--FC=mpiifx \
--CXX=mpiicpx \
--with-debugging=0 \
--CFLAGS='-O3 -qopenmp -qmkl -xHost' \
--FFLAGS='-O3 -qopenmp -qmkl -xHost' \
--CXXFLAGS='-O3 -qopenmp -qmkl -xHost' \
--with-blaslapack-dir=${MKLROOT} \
--with-mkl_cpardiso-dir=${MKLROOT} \
--with-scalar-type=complex \
---with-openmp-kernels=1 \
--download-slepc=1 \
--download-hdf5=1 \
--download-metis=1 \
--download-parmetis=1 \
--download-superlu_dist=1 \
--download-sowing=1
