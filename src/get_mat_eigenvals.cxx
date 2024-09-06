#include "hdf5.h"
#include "matexpre.h"
#include "petscdm.h"
#include "petscsys.h"
#include "slepceps.h"
#include <complex>
#include <petsc.h>
#include <slepc.h>
#include <vector>

int main(int argc, char **argv) {

  PetscCall(
      SlepcInitialize(&argc, &argv, (char *)0, "Calculate eigenvalues of A"));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Hello world!\n"));

  std::vector<PetscReal> omega_list{10.0, 20.0, 30.0, 40.0, 50.0,
                                    60.0, 70.0, 80.0, 90.0, 100.0};
  PetscInt pts_per_wavelen = 10;

  // Create hdf5 database.
  hid_t file_id =
      H5Fcreate("results.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group_id = H5Gcreate(file_id, std::to_string(omega).c_str(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  for (auto omega : omega_list) {
    PetscReal interior_domain_lens[2] = {1.0, 1.0};
    PetscReal v_max = 1.0;
    auto wavelength = v_max * 2.0 * PETSC_PI / omega;
    auto h = wavelength / pts_per_wavelen;
    PetscReal absorber_ratio = 0.25;
    PetscInt interior_elems[2] = {
        static_cast<PetscInt>(interior_domain_lens[0] / h) + 1,
        static_cast<PetscInt>(interior_domain_lens[1] / h) + 1};
    PetscInt absorber_elems[2] = {
        static_cast<PetscInt>(absorber_ratio * interior_elems[0]) + 1,
        static_cast<PetscInt>(absorber_ratio * interior_elems[1]) + 1};
    MatExpre<2u> matexpre(interior_domain_lens, interior_elems, absorber_elems,
                          omega, 25.0);
    PetscCall(matexpre.print_info());

    // Create velocity vector and matrix.
    Vec velocity = nullptr;
    PetscCall(DMCreateGlobalVector(matexpre.dm, &velocity));
    PetscCall(VecSet(velocity, v_max));
    Mat A = nullptr;
    PetscCall(DMCreateMatrix(matexpre.dm, &A));
    PetscCall(matexpre.get_mat(velocity, A));

    EPS eps = nullptr;
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

    std::vector<std::complex<PetscScalar>> eigvals(nconv);

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

    PetscCall(EPSDestroy(&eps));

    // Distroy velocity vector and matrix.
    PetscCall(VecDestroy(&velocity));
    PetscCall(MatDestroy(&A));

    // Write eigenvalues to hdf5 database.
  }

  // Close hdf5 database.
  auto status = H5Fclose(file_id);

  PetscCall(SlepcFinalize());
  return 0;
}