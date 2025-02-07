#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "slepceps.h"
#include "solver.h"
#include <vector>

std::complex<double> func_two_pole(const double x, const double y,
                                   const double z, void *ctx) {
  double r = *reinterpret_cast<double *>(ctx);
  GaussianCtx m_pole = {{0.5 - r, 0.5, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx p_pole = {{0.5 + r, 0.5, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  return -func_gaussian(x, y, z, &m_pole) + func_gaussian(x, y, z, &p_pole);
}

std::complex<double> func_four_pole(const double x, const double y,
                                    const double z, void *ctx) {
  double r = *reinterpret_cast<double *>(ctx);
  GaussianCtx mm_pole = {{0.5 - r, 0.5 - r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx pp_pole = {{0.5 + r, 0.5 + r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx mp_pole = {{0.5 - r, 0.5 + r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx pm_pole = {{0.5 + r, 0.5 - r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};

  return func_gaussian(x, y, z, &mm_pole) + func_gaussian(x, y, z, &pp_pole) -
         func_gaussian(x, y, z, &mp_pole) - func_gaussian(x, y, z, &pm_pole);
}

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  // Data need to be cleaned up.
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr, residual = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;

    // q = 10, ratio = 15/16, k = 3 * z^(l-5)
    // The grid is 2^l, omega is 2 * pi * k.

    PetscInt pts_per_wavelen = 10;
    PetscInt grids = 6;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-grids", &grids, nullptr));
    PetscCheck(grids >= 5, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "grids must be at least 5, but got %d.", grids);
    PetscInt k = 3 * 1 << (grids - 5);
    double ratio = 15.0 / 16;

    PetscBool use_csp = PETSC_FALSE, use_matex = PETSC_FALSE;
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_csp", &use_csp, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex", &use_matex,
                                  nullptr));

    // Update omega through k.
    PetscReal omega = 2.0 * PETSC_PI * k;

    // "solver" will be automatically cleaned up after the scope.
    Solver<2> solver(grids, ratio);

    // Create velocity vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &velocity));
    PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    // Create source vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &source));
    PetscBool use_four_pole = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_four_pole",
                                  &use_four_pole, nullptr));
    if (use_four_pole) {
      double r = 3.0 / (1 << grids);
      PetscCall(solver.get_vec_from_func(source, func_four_pole, &r));
    } else {
      double r = 3.0 / (1 << grids);
      PetscCall(solver.get_vec_from_func(source, func_two_pole, &r));
    }
    // PetscCall(solver.get_delta_rhs(source));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));
    // Create matrix.
    PetscCall(DMCreateMatrix(solver.dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    PetscCall(get_shifted_velocity_mat(A, velocity, -omega * omega));
    // Create solution vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    // Set the default ksp solver.
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPSetFromOptions(ksp));
    // PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    if (use_csp) {
      ComplexShiftPre csp_ctx = {0.1, omega, velocity, nullptr, nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_ComplexShiftPre(pc, &csp_ctx));
    }
    if (use_matex) {
      MatExPre matex_ctx = {0.1, 5, omega, omega, velocity, nullptr, nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPre(pc, &matex_ctx));
    }
    PetscCall(KSPSetUp(ksp));
    // Get info.
    PetscCall(solver.print_info(omega));
    // Zero the boundary.
    PetscCall(solver.get_zeroed_boundary_vec(source));
    PetscCall(KSPSolve(ksp, source, u));
    PetscCall(KSPConvergedReasonView(ksp, nullptr));

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

    // Save the source and the solution.
    PetscBool save_file = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-save_file", &save_file,
                                  nullptr));
    if (!save_file) {
      std::string surfix("pole");
      surfix += use_four_pole ? "4" : "2";
      surfix += "_grids" + std::to_string(grids);
      PetscCall(solver.save_xdmf_hdf5(source, surfix.c_str(), "data.hdf5",
                                      surfix.c_str()));
      PetscCall(solver.save_xdmf_hdf5(u, surfix.c_str(), "data.hdf5",
                                      surfix.c_str()));
    }

    // Mat v_2_A = nullptr;
    // EPS eps = nullptr;
    // // Study the eigenvalues.
    // PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &v_2_A));
    // // Borrow the residual vector.
    // PetscCall(VecPointwiseMult(residual, velocity, velocity));
    // PetscCall(MatDiagonalScale(v_2_A, residual, nullptr));
    // // Slepc stuff.
    // PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    // PetscCall(EPSSetOperators(eps, A, nullptr));
    // PetscCall(EPSSetProblemType(eps, EPS_NHEP));
    // PetscCall(EPSSetDimensions(eps, 8, PETSC_DEFAULT, PETSC_DEFAULT));
    // PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY));
    // PetscCall(EPSSetFromOptions(eps));
    // PetscCall(EPSSolve(eps));
    // PetscInt nconv = 0;
    // PetscCall(EPSGetConverged(eps, &nconv));
    // PetscPrintf(PETSC_COMM_WORLD, "Number of converged eigenpairs: %d\n",
    //             nconv);
    // std::vector<PetscScalar> eigvals(nconv);
    // for (PetscInt i = 0; i < nconv; ++i) {
    //   PetscScalar kr = 0.0 + 0.0i;
    //   PetscReal lambda_r = 0.0, lambda_i = 0.0;

    //   PetscCall(EPSGetEigenpair(eps, i, &kr, nullptr, nullptr, nullptr));
    //   lambda_r = PetscRealPart(kr);
    //   lambda_i = PetscImaginaryPart(kr);
    //   PetscPrintf(PETSC_COMM_WORLD, "Eigenvalue %d: %.5e\t+\t%.5ei\n", i,
    //               lambda_r, lambda_i);

    //   eigvals[i] = kr;
    // }
    // PetscCall(EPSDestroy(&eps));
    // PetscCall(MatDestroy(&v_2_A));

    // Clean up.
    PetscCall(DMRestoreGlobalVector(solver.dm, &residual));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(SlepcFinalize());
}