#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "solver.h"

std::complex<double> func_two_pole(const double x, const double y,
                                   const double z, void *ctx) {
  double r = *reinterpret_cast<double *>(ctx);
  GaussianCtx m_pole = {{0.5 - 2.0 * r, 0.5, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx p_pole = {{0.5 + 2.0 * r, 0.5, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  return -func_gaussian(x, y, z, &m_pole) + func_gaussian(x, y, z, &p_pole);
}

std::complex<double> func_four_pole(const double x, const double y,
                                    const double z, void *ctx) {
  double r = *reinterpret_cast<double *>(ctx);
  GaussianCtx mm_pole = {
      {0.5 - 2.0 * r, 0.5 - 2.0 * r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx pp_pole = {
      {0.5 + 2.0 * r, 0.5 + 2.0 * r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx mp_pole = {
      {0.5 - 2.0 * r, 0.5 + 2.0 * r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};
  GaussianCtx pm_pole = {
      {0.5 + 2.0 * r, 0.5 - 2.0 * r, 0.0}, r, 0.5 / (r * r * PETSC_PI)};

  return func_gaussian(x, y, z, &mm_pole) + func_gaussian(x, y, z, &pp_pole) -
         func_gaussian(x, y, z, &mp_pole) - func_gaussian(x, y, z, &pm_pole);
}

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr, residual = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;

    PetscInt pts_per_wavelen = 10;
    PetscInt freq = 20;
    PetscInt pml_width = 1;
    double omega = -1.0;
    PetscBool use_csp = PETSC_FALSE, use_matex = PETSC_FALSE;

    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pts_per_wavelen",
                                 &pts_per_wavelen, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-freq", &freq, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pml_width", &pml_width,
                                 nullptr));
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_csp", &use_csp, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex", &use_matex,
                                  nullptr));

    omega = 2.0 * PETSC_PI * freq;

    Solver<2> solver(freq * pts_per_wavelen, pml_width * pts_per_wavelen);
    // Create velocity vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &velocity));
    PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));

    // Create source vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &source));
    // Radius is 3h.
    double r = 3.0 / (pts_per_wavelen * freq);
    PetscBool use_four_pole = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_four_pole",
                                  &use_four_pole, nullptr));
    if (use_four_pole) {
      PetscCall(solver.get_vec_from_func(source, func_four_pole, &r));
    } else {
      PetscCall(solver.get_vec_from_func(source, func_two_pole, &r));
    }
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));

    // Create matrix.
    PetscCall(DMCreateMatrix(solver.dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    // Now, A is -Delta, and we need A = omega^2 Id + v^2 Delta,
    // such that Im(lambda(A)) >= 0.
    // Borrow residual.
    PetscCall(DMGetGlobalVector(solver.dm, &residual));
    PetscCall(VecPointwiseDivide(residual, velocity, velocity));
    PetscCall(MatDiagonalScale(A, residual, nullptr));
    PetscCall(MatShift(A, -omega * omega));
    PetscCall(MatScale(A, -1.0));
    // Create solution vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    // Set the default ksp solver.
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPSetFromOptions(ksp));
    if (use_csp) {
      ComplexShiftPre csp_ctx = {1.0 + 0.1i, omega, velocity, nullptr, nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_ComplexShiftPre(pc, &csp_ctx));
    }
    if (use_matex) {
      MatExPre matex_ctx = {1.0 / (omega * omega), freq, nullptr, nullptr,
                            nullptr};
      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPre(pc, &matex_ctx));
    }
    PetscCall(KSPSetUp(ksp));

    // Get info and solve.
    PetscCall(solver.print_info(omega));
    PetscCall(KSPSolve(ksp, source, u));
    PetscCall(KSPConvergedReasonView(ksp, nullptr));

    // Save the source and the solution.
    PetscBool save_file = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-save_file", &save_file,
                                  nullptr));
    if (save_file) {
      std::string surfix("pole");
      surfix += use_four_pole ? "4" : "2";
      surfix += "_freq" + std::to_string(freq);
      PetscCall(solver.save_xdmf_hdf5(source, surfix.c_str(), "data.hdf5",
                                      surfix.c_str()));
      PetscCall(solver.save_xdmf_hdf5(u, surfix.c_str(), "data.hdf5",
                                      surfix.c_str()));
    }

    // Mannually check the resiudal again.
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

    PetscCall(DMRestoreGlobalVector(solver.dm, &residual));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }
  PetscCall(SlepcFinalize());
}