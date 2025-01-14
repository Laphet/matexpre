#include "petscdm.h"
#include "petscmat.h"
#include "petscvec.h"
#include "slepceps.h"
#include "solver.h"
#include <vector>

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
  {
    // Data need to be cleaned up.
    Vec velocity = nullptr, source = nullptr, u = nullptr, residual = nullptr;
    Mat A = nullptr, v_minus2_A = nullptr;
    KSP ksp = nullptr;
    EPS eps = nullptr;

    PetscInt pts_per_wavelen = 10;
    PetscInt k = 20;
    PetscInt absorber_elems = 10;

    // Get options from command line.
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-pts_per_wavelen",
                                 &pts_per_wavelen, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-k", &k, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-absorber_elems",
                                 &absorber_elems, nullptr));

    // Update omega through k.
    double omega = 2.0 * PETSC_PI * k;

    // "solver" will be automatically cleaned up after the scope.
    Solver<2> solver(pts_per_wavelen * k, absorber_elems);

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
    // Create the Laplace matrix.
    PetscCall(DMCreateMatrix(solver.dm, &A));
    PetscCall(solver.get_laplace_mat(A, omega));
    // Create the solution vector.
    PetscCall(DMCreateGlobalVector(solver.dm, &u));
    PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(u), "solution"));

    // Solve the system.
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp, source, u));
    PetscCall(KSPConvergedReasonView(ksp, nullptr));

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

    // Study the eigenvalues.
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &v_minus2_A));
    // Borrow the residual vector.
    PetscCall(VecPointwiseMult(residual, velocity, velocity));
    PetscCall(VecReciprocal(residual));
    PetscCall(MatDiagonalScale(v_minus2_A, residual, nullptr));
    // Slepc stuff.
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, A, nullptr));
    PetscCall(EPSSetProblemType(eps, EPS_NHEP));
    PetscCall(EPSSetDimensions(eps, 8, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscInt nconv = 0;
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscPrintf(PETSC_COMM_WORLD, "Number of converged eigenpairs: %d\n",
                nconv);
    std::vector<PetscScalar> eigvals(nconv);
    for (PetscInt i = 0; i < nconv; ++i) {
      PetscScalar kr = 0.0 + 0.0i;
      PetscReal lambda_r = 0.0, lambda_i = 0.0;

      PetscCall(EPSGetEigenpair(eps, i, &kr, nullptr, nullptr, nullptr));
      lambda_r = PetscRealPart(kr);
      lambda_i = PetscImaginaryPart(kr);
      PetscPrintf(PETSC_COMM_WORLD, "Eigenvalue %d: %.5e\t+\t%.5ei\n", i,
                  lambda_r, lambda_i);

      eigvals[i] = kr;
    }

    // Clean up.
    PetscCall(EPSDestroy(&eps));
    PetscCall(DMRestoreGlobalVector(solver.dm, &residual));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&velocity));
  }

  PetscCall(SlepcFinalize());
}