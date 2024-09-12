#include "petsc.h"
#include "petscerror.h"
#include "petsclog.h"
#include "petscsys.h"
#include <iterator>
#include <sstream>
#include <vector>

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

  // DM dm = nullptr;
  // PetscInt nx = 8, ny = 9;
  // PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,
  // DM_BOUNDARY_NONE,
  //                        DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE,
  //                        PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm));

  // PetscCall(DMSetUp(dm));

  PetscInt mpi_size = -1, mpi_rank = -1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank));

  // PetscInt local_n0 = ny / mpi_size + (ny % mpi_size > mpi_rank ? 1 : 0);
  // PetscInt local_0_start =
  //     mpi_rank * (ny / mpi_size) + PetscMin(mpi_rank, ny % mpi_size);

  // PetscCall(PetscSynchronizedPrintf(
  //     PETSC_COMM_SELF,
  //     "mpi_size=%d, mpi_rank=%d, local_n0=%d, local_0_start=%d\n", mpi_size,
  //     mpi_rank, local_n0, local_0_start));
  // PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // MatStencil lower = {0, 0, 0, 0}, upper = {0, 0, nx, 0};
  // lower.j = local_0_start;
  // upper.j = local_0_start + local_n0;

  // IS fftw_patch_is = nullptr;
  // PetscCall(DMDACreatePatchIS(dm, &lower, &upper, &fftw_patch_is,
  // PETSC_TRUE));

  // PetscInt batch_size = 0;
  // PetscCall(ISGetLocalSize(fftw_patch_is, &batch_size));
  // PetscCall(PetscSynchronizedPrintf(
  //     PETSC_COMM_SELF, "mpi_rank=%d, batch_size=%d\n", mpi_rank,
  //     batch_size));
  // PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // PetscCall(ISView(fftw_patch_is, PETSC_VIEWER_STDOUT_SELF));
  // PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF));

  // PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  // Vec v_global = nullptr;
  // PetscCall(DMCreateGlobalVector(dm, &v_global));
  // PetscCall(VecSet(v_global, PETSC_PI));

  // Vec v_local = nullptr;
  // PetscCall(VecCreateSeq(PETSC_COMM_SELF, batch_size, &v_local));

  // VecScatter scatter = nullptr;
  // PetscCall(
  //     VecScatterCreate(v_global, fftw_patch_is, v_local, nullptr, &scatter));

  // PetscCall(VecScatterBegin(scatter, v_global, v_local, INSERT_VALUES,
  //                           SCATTER_FORWARD));
  // PetscCall(VecScatterEnd(scatter, v_global, v_local, INSERT_VALUES,
  //                         SCATTER_FORWARD));
  // PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF));

  // PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  // PetscCall(VecView(v_local, PETSC_VIEWER_STDOUT_SELF));
  // PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF));

  // PetscCall(VecSet(v_local, mpi_rank));
  // PetscCall(VecView(v_local, PETSC_VIEWER_STDOUT_SELF));
  // PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF));
  // PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  // PetscCall(VecScatterBegin(scatter, v_global, v_local, INSERT_VALUES,
  //                           SCATTER_REVERSE));
  // PetscCall(VecScatterEnd(scatter, v_global, v_local, INSERT_VALUES,
  //                         SCATTER_REVERSE));

  // PetscCall(VecView(v_global, PETSC_VIEWER_STDOUT_SELF));
  // PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF));

  // PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  // PetscCall(VecScatterDestroy(&scatter));
  // PetscCall(VecDestroy(&v_global));
  // PetscCall(ISDestroy(&fftw_patch_is));
  // PetscCall(DMDestroy(&dm));

  std::vector<PetscInt> send_buf(mpi_size, 0);
  std::vector<PetscInt> recv_buf(mpi_size, -1);
  // for (auto i = 0; i < mpi_size; ++i)
  //   send_buf[i] = i;
  send_buf[mpi_rank] = mpi_rank * mpi_rank;
  PetscCallMPI(MPI_Allgather(&send_buf[mpi_rank], 1, MPIU_INT, &recv_buf[0], 1,
                             MPIU_INT, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  std::ostringstream oss;
  std::copy(recv_buf.begin(), recv_buf.end(),
            std::ostream_iterator<int>(oss, " "));
  PetscPrintf(PETSC_COMM_SELF, "rank=%d, %s\n", mpi_rank, oss.str().c_str());

  PetscCall(PetscFinalize());
  return 0;
}