# Install Petsc
Assume intel-oneapi-mpi and mkl are proper, which means that you can correctly find mpiicc, mpiifort, mpiicpc, and ${MKLROOT}

./configure \
PETSC_ARCH=linux-oneapi-opt \
--CC=mpiicc \
--FC=mpiifort \
--CXX=mpiicpc \
--with-debugging=0 \
--CFLAGS='-O3 -qopenmp -qmkl -xhost' \
--FFLAGS='-O3 -qopenmp -qmkl -xhost' \
--CXXFLAGS='-O3 -qopenmp -qmkl -xhost' \
--with-blaslapack-dir=${MKLROOT} \
--with-mkl_cpardiso-dir=${MKLROOT} \
--download-scalapack=1 \
--with-scalar-type=complex \
---with-openmp-kernels=1 \
--download-slepc=1 \
--download-mumps=1 \
--download-suitesparse=1 \
--download-hdf5=1 \
--download-sowing=1 \
--download-metis=1 \
--download-parmetis=1 \
--download-superlu_dist=1 \
--download-hypre=1