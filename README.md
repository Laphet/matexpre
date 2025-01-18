# Install Petsc
Assume intel-oneapi-mpi and mkl are proper, which means that you can correctly find mpiicc(mpiicx), mpiifort(mpiifx), mpiicpc(mpiicpx), and ${MKLROOT}.
The project may need slepc, hdf5, and one mpi direct solver (superLU_dist, mumps, cpardiso).
Try the follow command to build petsc, and you may specify correct optimization flags regarding the CPU architecture.

./configure \
PETSC_ARCH=linux-oneapi-complex-opt \
--CC=mpiicc \
--FC=mpiifort \
--CXX=mpiicpc \
--with-debugging=0 \
--CFLAGS='-O3 -qopenmp -qmkl -axCORE-AVX512' \
--FFLAGS='-O3 -qopenmp -qmkl -axCORE-AVX512' \
--CXXFLAGS='-O3 -qopenmp -qmkl -axCORE-AVX512' \
--with-blaslapack-dir=${MKLROOT} \
--with-mkl_cpardiso-dir=${MKLROOT} \
--with-scalar-type=complex \
---with-openmp-kernels=1 \
--download-slepc=downloaded_packages/slepc-v3.21.1.tar.gz \
--download-hdf5=downloaded_packages/hdf5-1.14.3-p1.tar.bz2 \
--download-metis=downloaded_packages/petsc-pkg-metis-69fb26dd0428.tar.gz \
--download-parmetis=downloaded_packages/petsc-pkg-parmetis-f5e3aab04fd5.tar.gz \
--download-superlu_dist=downloaded_packages/superlu_dist-8.2.1.tar.gz \
--download-sowing=downloaded_packages/petsc-pkg-sowing-221cd70e0f28.tar.gz
