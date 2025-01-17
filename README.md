# Install Petsc
Assume intel-oneapi-mpi and mkl are proper, which means that you can correctly find mpiicc, mpiifort, mpiicpc, and ${MKLROOT}.
The project may need slepc, hdf5, and one mpi direct solver (superLU_dist, mumps, cpardiso).
Try the follow command from scratch.


./configure \
PETSC_ARCH=linux-oneapi-opt \
--CC=mpiicc \
--FC=mpiifort \
--CXX=mpiicpc \
--with-debugging=0 \
--CFLAGS='-O3 -qopenmp -xhost' \
--FFLAGS='-O3 -qopenmp -xhost' \
--CXXFLAGS='-O3 -qopenmp -xhost' \
--with-blaslapack-dir=${MKLROOT} \
--with-mkl_cpardiso-dir=${MKLROOT} \
--download-scalapack=1 \
--with-scalar-type=complex \
---with-openmp-kernels=1 \
--download-sowing=1 \
--download-slepc=1 \
--download-hdf5=1 \
--download-metis=1 \
--download-parmetis=1 \
--download-superlu_dist=1 
