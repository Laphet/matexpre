#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
// #include "slepceps.h"
#include "solver.h"

std::complex<double> conv_lens_velocity(const double x, const double y,
                                        const double z, void *ctx) {
  double r = *reinterpret_cast<double *>(ctx);
  GaussianCtx pole = {{0.5, 0.1, 0.0}, r, 0.5};
  return 1.0 - func_gaussian(x, y, z, &pole);
}

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;
    DM dm = nullptr;

    PetscInt pts_per_wavelen = 10;
    PetscInt freq = 20;
    PetscInt pml_width = 10;
    double omega = -1.0;

    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pts_per_wavelen",
                                 &pts_per_wavelen, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-freq", &freq, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pml_width", &pml_width,
                                 nullptr));

    omega = 2.0 * PETSC_PI * freq;

    // Because v_min=0.5, h should be smaller.
    Solver<2> solver(freq * pts_per_wavelen * 2, pml_width);
    // Borrow the DM.
    PetscCall(solver.get_dm(&dm));
    // Create velocity vector.
    PetscCall(DMCreateGlobalVector(dm, &velocity));
    double r = 1.0 / 32.0;
    PetscCall(solver.get_vec_from_func(velocity, conv_lens_velocity,
                                       reinterpret_cast<void *>(&r)));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));

    // Create source vector.
    PetscCall(DMCreateGlobalVector(dm, &source));
    // The source is at the center of the domain.
    PetscCall(solver.get_delta_rhs(source));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));

    // Create matrix.
    PetscCall(DMCreateMatrix(dm, &A));
    PetscBool use_pml = PETSC_FALSE;
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_pml", &use_pml, nullptr));
    if (use_pml) {
      PetscCall(solver.get_laplace_pml_mat(A, omega));
    } else {
      PetscCall(solver.get_laplace_abc_mat(A, omega));
    }
    // { PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD)); }
    PetscCall(get_shifted_velocity_mat(A, velocity, -omega * omega));
    PetscCall(MatScale(A, -1.0));
    // Create solution vector.
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    // Set the default ksp solver.
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPSetFromOptions(ksp));
    PetscBool use_csp = PETSC_FALSE, use_matex = PETSC_FALSE,
              use_matex_ver2 = PETSC_FALSE, use_matex_ver3 = PETSC_FALSE,
              use_matex_mg = PETSC_FALSE;
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_csp", &use_csp, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex", &use_matex,
                                  nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex_ver2",
                                  &use_matex_ver2, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex_ver3",
                                  &use_matex_ver3, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex_mg",
                                  &use_matex_mg, nullptr));

    ComplexShiftPre csp_ctx;
    if (use_csp) {
      csp_ctx.shift = -1.0;
      csp_ctx.omega = omega;
      csp_ctx.velocity = velocity;
      csp_ctx.matex_ctx.delta_t = -1.0;
      csp_ctx.matex_ctx.steps = -1;

      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_ComplexShiftPre(pc, &csp_ctx));
    }

    MatExPre matexpre_ctx;
    if (use_matex) {
      matexpre_ctx.delta_t = -1.0;
      matexpre_ctx.steps = -1;

      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPre(pc, &matexpre_ctx));
    }

    MatExPreVer2Ctx matexprever2_ctx;
    if (use_matex_ver2) {
      matexprever2_ctx.delta_t = -1.0;
      matexprever2_ctx.shift = -1.0;
      matexprever2_ctx.omega = omega;
      matexprever2_ctx.velocity = velocity;

      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPreVer2(pc, &matexprever2_ctx));
    }

    MatExPreVer3Ctx matexprever3_ctx;
    if (use_matex_ver3) {
      matexprever3_ctx.delta_t = -1.0;
      matexprever3_ctx.steps = -1;
      matexprever3_ctx.shift = -1.0;
      matexprever3_ctx.omega = omega;
      matexprever3_ctx.velocity = velocity;

      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPreVer3(pc, &matexprever3_ctx));
    }

    MatExPreMg matexpremg_ctx;
    if (use_matex_mg) {
      matexpremg_ctx.omega = omega;
      matexpremg_ctx.velocity = velocity;
      matexpremg_ctx.delta_t = -1.0;
      matexpremg_ctx.shift = -1.0;

      PC pc = nullptr;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCShell_MatExPreMg(pc, &matexpremg_ctx));
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
      std::string surfix("conv_lens");
      surfix += "_freq" + std::to_string(freq);
      PetscCall(solver.save_xdmf_hdf5(u, surfix.c_str(), "data_conv_lens.hdf5",
                                      surfix.c_str()));
      PetscCall(solver.save_xdmf_hdf5(velocity, surfix.c_str(),
                                      "data_conv_lens.hdf5", surfix.c_str()));
    }

    {
      // Set up the eigenvalue problem.
      // EPS eps = nullptr;
      // PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
      // PetscCall(EPSSetOperators(eps, A, nullptr));
      // PetscCall(EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT));
      // PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_IMAGINARY));
      // PetscCall(EPSSetFromOptions(eps));
      // PetscCall(EPSSolve(eps));
      // PetscInt nconv = 0;
      // PetscCall(EPSGetConverged(eps, &nconv));
      // PetscPrintf(PETSC_COMM_WORLD, "Number of converged eigenpairs: %d\n",
      //             nconv);
      // for (PetscInt i = 0; i < nconv; ++i) {
      //   PetscScalar kr = 0.0 + 0.0 * IU;
      //   PetscCall(EPSGetEigenpair(eps, i, &kr, nullptr, nullptr, nullptr));
      //   auto lambda_r = PetscRealPart(kr);
      //   auto lambda_i = PetscImaginaryPart(kr);
      //   PetscPrintf(PETSC_COMM_WORLD, "Eigenvalue %d: %.5e\t+\t%.5ei, ", i,
      //               lambda_r, lambda_i);
      //   double scaled_val = 0.0;
      //   scaled_val = std::abs(lambda_i);
      //   scaled_val /= omega * omega;
      //   PetscPrintf(PETSC_COMM_WORLD, "scaled value: %.5e\n", scaled_val);
      // }
      // PetscCall(EPSDestroy(&eps));
    }

    // Clean up.
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }
  PetscCall(SlepcFinalize());
}