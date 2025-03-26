#include "petscdm.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "slepcsys.h"
#include "solver.h"
#include <string>

const int NX = 801;
const int NY = 801;
const int NZ = 187;
const double LX = 20.0;        // km
const double LY = 20.0;        // km
const double LZ = 4.65;        // km
const double VMIN = 2178.8345; // m/s
const double VMAX = 6.0;
const double MAX_FREQ = 80.0;
char HDF5_FILENAME[] = "data_overthrust.hdf5";
char HDF5_GROUPNAME[] = "overthrust";
char VELOCITY_NAME[] = "velocity";

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  // Read the Marmousi model.
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;

    int pml_width = 10;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pml_width", &pml_width,
                                 nullptr));
    PetscReal freq = MAX_FREQ;
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-freq", &freq, nullptr));

    // Copy the velocity into the solver dm.
    int interior_elems[3] = {NX - 1, NY - 1, NZ - 1};
    // double marmousi_interior_domain_lens[2] = {MARMOUSI_LX, MARMOUSI_LY};
    double interior_domain_lens[3] = {1.0, LY / LX, LZ / LX};
    Solver<3> solver(pml_width, interior_elems, interior_domain_lens);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reading the velocity model>>>\n"));

    // Create the velocity vector.
    DM dm = nullptr;
    PetscCall(solver.get_dm(&dm));
    PetscCall(DMCreateGlobalVector(dm, &velocity));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    PetscCall(solver.read_hdf5_vec(velocity, HDF5_FILENAME, HDF5_GROUPNAME,
                                   VELOCITY_NAME, 1.0 / VMIN));
    // PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reading the velocity model<<<\n"));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Preparing the system>>>\n"));
    // Source is at the center.
    PetscCall(DMCreateGlobalVector(dm, &source));
    PetscCall(solver.get_delta_rhs(source));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));

    // Form the system.
    double omega = 2.0 * PETSC_PI * freq;
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));
    PetscCall(DMCreateMatrix(dm, &A));
    PetscBool use_pml = PETSC_FALSE;
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_pml", &use_pml, nullptr));
    if (use_pml) {
      PetscCall(solver.get_laplace_pml_mat(A, omega));
    } else {
      PetscCall(solver.get_laplace_abc_sim_mat(A, omega));
    }
    PetscCall(get_shifted_velocity_mat(A, velocity, -omega * omega));
    PetscCall(MatScale(A, -1.0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Preparing the system<<<\n"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
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
    // Set the default ksp solver.
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPSetUp(ksp));

    // Get info and solve.
    PetscCall(solver.print_info(omega));
    PetscCall(KSPSolve(ksp, source, u));
    PetscCall(KSPConvergedReasonView(ksp, nullptr));

    // Save vectors.
    PetscBool save_file = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-save_file", &save_file,
                                  nullptr));
    if (save_file) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Save the files>>>\n"));
      std::string surfix(HDF5_GROUPNAME);
      surfix += std::string("_f80w") + std::to_string(pml_width);
      PetscCall(solver.save_xdmf_hdf5(velocity, surfix.c_str(), HDF5_FILENAME,
                                      surfix.c_str()));
      PetscCall(solver.save_xdmf_hdf5(u, surfix.c_str(), HDF5_FILENAME,
                                      surfix.c_str()));
      // PetscCall(solver.save_bin(velocity, surfix.c_str()));
      // PetscCall(solver.save_bin(u, surfix.c_str()));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Save the files<<<\n"));
    }

    // Clean up.
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(SlepcFinalize());
}