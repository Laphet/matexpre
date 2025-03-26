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
--with-openmp-kernels=1 \
--with-openmp \
--download-slepc=externals/slepc-e800285704065475b64d80f654e104905f7c84bb.tar.gz \
--download-hdf5=externals/hdf5-1.14.3-p1.tar.bz2 \
--download-sowing=externals/petsc-pkg-sowing-8ec17636b4e9.tar.gz \
--download-metis=externals/petsc-pkg-metis-69fb26dd0428.tar.gz \
--download-parmetis=externals/petsc-pkg-parmetis-f5e3aab04fd5.tar.gz \
--download-superlu_dist=externals/superlu_dist-8.2.1.tar.gz \
--with-packages-download-dir=externals/ \
