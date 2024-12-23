#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscsys.h"
#include "petscvec.h"
#include "solver.h"

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  // Data need to be cleaned up.
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr, residual = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;

    PetscInt pts_per_wavelen = 10;
    PetscInt k = 20;
    double omega = 2.0 * PETSC_PI * k;

    // "solver" will be automatically cleaned up after the scope.
    Solver<2> solver(pts_per_wavelen * k, 10);

    // Create velocity vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &velocity));
    PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    // Create source vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &source));
    GaussianCtx ctx = {{0.5, 0.2, 0.0}, 0.1, 1.0};
    PetscCall(solver.get_vec_from_func(source, func_gaussian, &ctx));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));
    // Create matrix.
    PetscCall(DMCreateMatrix(solver.dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    PetscCall(solver.get_final_mat(A, velocity, omega));
    // Create solution vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    // PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp, source, u));

    // Get info.
    PetscCall(solver.print_info());
    PetscInt its = -1;
    PetscCall(KSPGetIterationNumber(ksp, &its));
    // PETSc convergence test should be ||P^{-1}(b - A x)|| < rtol ||P^{-1}b||,
    // which is not residual l2 norm.
    // This is reasonalbe because P^{-1}b has the same unit as u.
    PetscCall(DMGetGlobalVector(solver.dm, &residual));
    PetscCall(MatResidual(A, source, u, residual));
    PetscReal source_norm = 0.0, residual_norm = 0.0;
    PetscCall(VecNorm(source, NORM_2, &source_norm));
    PetscCall(VecNorm(residual, NORM_2, &residual_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Number of iterations=%d, relative residual "
                          "norm=%.5e, source norm=%.5e, residual norm=%.5e.\n",
                          its, residual_norm / source_norm, source_norm,
                          residual_norm));

    PetscCall(solver.save_xdmf_hdf5(u, "-test", "data.hdf5", "pml_solver"));

    // Clean up.
    PetscCall(DMRestoreGlobalVector(solver.dm, &residual));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(PetscFinalize());
}