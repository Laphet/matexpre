#include "petscdm.h"
#include "petscmat.h"
#include "petscvec.h"
#include "slepceps.h"
#include "solver.h"

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  {
    // Data need to be cleaned up.
    Vec velocity = nullptr, velocity_sq = nullptr;
    Mat A = nullptr;
    KSP ksp = nullptr;
    EPS eps = nullptr;

    PetscInt pts_per_wavelen = 10;
    PetscInt freq = 20;
    PetscInt pml_width = 1;

    // Get options from command line.
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pts_per_wavelen",
                                 &pts_per_wavelen, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-freq", &freq, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pml_width", &pml_width,
                                 nullptr));

    // Update omega through k.
    double omega = 2.0 * PETSC_PI * freq;

    // "solver" will be automatically cleaned up after the scope.
    Solver<2> solver(pts_per_wavelen * freq, pml_width * pts_per_wavelen);
    PetscCall(solver.print_info(omega));

    // Create velocity vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &velocity));
    PetscCall(solver.get_vec_from_func(velocity, func_one, nullptr));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(velocity),
                                 "velocity"));
    // Create the Laplace matrix.
    PetscCall(DMCreateMatrix(solver.dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    // Now, A = -v^2 Delta
    PetscCall(DMGetGlobalVector(solver.dm, &velocity_sq));
    PetscCall(VecPointwiseMult(velocity_sq, velocity, velocity));
    PetscCall(MatDiagonalScale(A, velocity_sq, nullptr));

    // Slepc stuff.
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, A, nullptr));
    PetscCall(EPSSetProblemType(eps, EPS_NHEP));
    PetscCall(EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscBool check_inv = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-check_inv", &check_inv,
                                  nullptr));
    if (check_inv) {
      PetscScalar target = omega * omega;
      PetscCall(EPSSetTarget(eps, target));
      PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
    } else {
      PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY));
    }

    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscInt nconv = 0;
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscPrintf(PETSC_COMM_WORLD, "Number of converged eigenpairs: %d\n",
                nconv);
    for (PetscInt i = 0; i < nconv; ++i) {
      PetscScalar kr = 0.0 + 0.0i;

      PetscCall(EPSGetEigenpair(eps, i, &kr, nullptr, nullptr, nullptr));
      auto lambda_r = PetscRealPart(kr);
      auto lambda_i = PetscImaginaryPart(kr);
      PetscPrintf(PETSC_COMM_WORLD, "Eigenvalue %d: %.5e\t+\t%.5ei, ", i,
                  lambda_r, lambda_i);
      double scaled_val = 0.0;
      if (check_inv) {
        scaled_val =
            std::sqrt((lambda_r - omega * omega) * (lambda_r - omega * omega) +
                      lambda_i * lambda_i);
        scaled_val /= omega;
      } else {
        scaled_val = std::abs(lambda_i);
        scaled_val /= omega * omega;
      }
      PetscPrintf(PETSC_COMM_WORLD, "scaled value: %.5e\n", scaled_val);
    }

    // Clean up.
    PetscCall(EPSDestroy(&eps));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(SlepcFinalize());
}