#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscsys.h"
#include "petscsystypes.h"
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
    // PetscInt absorber_elems = 10;
    double omega = -1.0, ratio = 0.92;
    PetscBool use_csp = PETSC_FALSE, use_matex = PETSC_FALSE;

    // Get options from command line.
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pts_per_wavelen",
                                 &pts_per_wavelen, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-k", &k, nullptr));
    // PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-absorber_elems",
    //                              &absorber_elems, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-ratio", &ratio, nullptr));
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_csp", &use_csp, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex", &use_matex,
                                  nullptr));

    // Update omega through k.
    omega = 2.0 * PETSC_PI * k;
    // Test omega=0.
    // omega = 0.0;

    PetscInt levels = std::ceil(std::log2(pts_per_wavelen * k / ratio));

    // "solver" will be automatically cleaned up after the scope.
    // Solver<2> solver(pts_per_wavelen * k, absorber_elems);
    Solver<2> solver(levels, ratio);

    // Create velocity vector.
    DM dm = nullptr;
    PetscCall(solver.get_dm(&dm));
    PetscCall(DMCreateGlobalVector(dm, &velocity));
    PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    // Create source vector.
    PetscCall(DMCreateGlobalVector(dm, &source));
    GaussianCtx ctx = {{0.5, 0.2, 0.0}, 0.1, 1.0};
    PetscCall(solver.get_vec_from_func(source, func_gaussian, &ctx));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));
    PetscCall(solver.get_zeroed_boundary_vec(source));
    // Create matrix.
    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    PetscCall(get_shifted_velocity_mat(A, velocity, -omega * omega));
    // PetscCall(PetscPrintf(
    //     PETSC_COMM_WORLD,
    //     "Test -i omega / v^2 Id - Laplace.\n Is this matrix easy to
    //     solve?\n"));
    // Create solution vector.
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    // Set the default ksp solver.
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPSetFromOptions(ksp));
    // PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    if (use_csp) {
      ComplexShiftPre csp_ctx = {1.0 + 0.1i, omega, velocity, nullptr, nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_ComplexShiftPre(pc, &csp_ctx));
    }
    if (use_matex) {
      MatExPre matex_ctx = {1.0 / (omega * omega), k, nullptr, nullptr,
                            nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPre(pc, &matex_ctx));
    }
    PetscCall(KSPSetUp(ksp));
    // Get info.
    PetscCall(solver.print_info(omega));
    PetscCall(KSPSolve(ksp, source, u));
    PetscCall(KSPConvergedReasonView(ksp, nullptr));

    PetscInt its = -1;
    PetscCall(KSPGetIterationNumber(ksp, &its));
    // PETSc convergence test should be ||P^{-1}(b - A x)|| < rtol ||P^{-1}b||,
    // which is not residual l2 norm.
    // This is reasonalbe because P^{-1}b has the same unit as u.
    PetscCall(DMGetGlobalVector(dm, &residual));
    PetscCall(MatResidual(A, source, u, residual));
    PetscReal source_norm = 0.0, residual_norm = 0.0;
    PetscCall(VecNorm(source, NORM_2, &source_norm));
    PetscCall(VecNorm(residual, NORM_2, &residual_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Number of iterations=%d, relative residual "
                          "norm=%.5e, source norm=%.5e, residual norm=%.5e.\n",
                          its, residual_norm / source_norm, source_norm,
                          residual_norm));

    // PetscCall(solver.save_xdmf_hdf5(u, "-test", "data.hdf5", "pml_solver"));

    // Clean up.
    PetscCall(DMRestoreGlobalVector(dm, &residual));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(PetscFinalize());
}

// Tests.
/*

mpiexec -n 16 ./main -k 20 -absorber_elems 10
  iter: 654
mpiexec -n 16 ./main -k 20 -absorber_elems 10 -use_csp -csp_ksp_max_it 1/2
  BREAKDOWN at iter=30
mpiexec -n 16 ./main -k 20 -absorber_elems 10 -use_csp -csp_ksp_max_it 1
-csp_shift 0.1
  BREAKDOWN at iter=30
mpiexec -n 16 ./main -k 20 -absorber_elems 10 -use_csp -csp_ksp_type preonly
-csp_shift 3.0
  Diverge at iter=10000
mpiexec -n 16 ./main -k 20 -absorber_elems 10 -pc_type gamg
  Diverge at iter=10000
mpiexec -n 16 ./main -k 20 -absorber_elems 10 -use_csp -csp_pc_type asm
  iter 9
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_csp -csp_pc_type asm
  iter 13
mpiexec -n 16 ./main -k 80 -absorber_elems 10 -use_csp -csp_pc_type asm
  iter 20
mpiexec -n 16 ./main -k 80 -absorber_elems 10 -pc_type asm
  iter 1421
mpiexec -n 16 ./main -k 80 -absorber_elems 10
  iter 1562
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_csp -csp_ksp_type preonly
-csp_pc_type lu -ksp_monitor_true_residual
iter 13

mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_matex -matex_ksp_type preonly
-matex_pc_type lu
*/