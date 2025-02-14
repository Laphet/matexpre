#include "petscdm.h"
#include "petscerror.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "slepcsys.h"
#include "solver.h"
#include <string>

constexpr int MARMOUSI_NX = 13601;
constexpr int MARMOUSI_NY = 2801;
constexpr double MARMOUSI_LX = 17.0;
constexpr double MARMOUSI_LY = 3.5;
constexpr double MARMOUSI_VMIN = 1.0;
char HDF5_FILENAME[] = "data.hdf5";
char HDF5_GROUPNAME[] = "marmousi-ii";
char P_VELOCITY_NAME[] = "P-velocity";

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  // Read the Marmousi model.
  {
    Vec velocity = nullptr, source = nullptr, u = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;

    // Three configurations.
    int config = 0, freq = 20, pts_per_wavelen = 40;
    PetscCall(
        PetscOptionsGetInt(nullptr, nullptr, "-config", &config, nullptr));
    switch (config) {
    case 1:
      freq = 40;
      pts_per_wavelen = 20;
      break;
    case 2:
      freq = 80;
      pts_per_wavelen = 10;
      break;
    default:
      break;
    }

    // Copy the velocity into the solver dm.
    int marmousi_interior_elems[2] = {MARMOUSI_NX - 1, MARMOUSI_NY - 1};
    double marmousi_interior_domain_lens[2] = {MARMOUSI_LX, MARMOUSI_LY};
    Solver<2> solver(pts_per_wavelen, marmousi_interior_elems,
                     marmousi_interior_domain_lens);

    // Create the velocity vector.
    DM dm = nullptr;
    PetscCall(solver.get_dm(&dm));
    PetscCall(DMCreateGlobalVector(dm, &velocity));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    PetscCall(solver.read_hdf5_vec(velocity, "data.hdf5", "marmousi-ii",
                                   "P-velocity"));

    // Source is at 10m-depth, hence delta source location should be j=8.
    PetscCall(DMCreateGlobalVector(dm, &source));
    PetscCall(solver.get_delta_rhs(source, MARMOUSI_NX / 2, 8, 0));
    PetscCall(
        PetscObjectSetName(reinterpret_cast<PetscObject>(source), "source"));

    // Form the system.
    double omega = 2.0 * PETSC_PI * freq;
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));
    // Borrow u, now u is v^2
    PetscCall(VecPointwiseMult(u, velocity, velocity));
    // Now A is -Delta, and we need A = omega^2 Id + v^2 Delta,
    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    PetscCall(MatDiagonalScale(A, u, nullptr));
    PetscCall(MatShift(A, -omega * omega));
    PetscCall(MatScale(A, -1.0));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscBool use_csp = PETSC_FALSE, use_matex = PETSC_FALSE;
    PetscCall(
        PetscOptionsGetBool(nullptr, nullptr, "-use_csp", &use_csp, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-use_matex", &use_matex,
                                  nullptr));
    if (use_csp) {
      ComplexShiftPre csp_ctx = {1.0 + 0.1 * IU, omega, velocity, nullptr,
                                 nullptr};
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
    PetscCall(KSPSetFromOptions(ksp));
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
      std::string surfix("marmousi-ii-config");
      surfix += std::to_string(config);
      PetscCall(solver.save_xdmf_hdf5(velocity, surfix.c_str(), HDF5_FILENAME,
                                      surfix.c_str()));
      PetscCall(solver.save_xdmf_hdf5(u, surfix.c_str(), HDF5_FILENAME,
                                      surfix.c_str()));
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