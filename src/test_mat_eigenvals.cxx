#include "hello_hdf5.h"
#include "matexpre.h"
#include "petscdm.h"
#include "petscsys.h"
#include "slepceps.h"
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <cstddef>
#include <petsc.h>
#include <slepc.h>
#include <string>
#include <vector>

const std::string HDF5_DB("data.hdf5");
const std::string ROOT_GROUP("eigenvalues");

int main(int argc, char **argv) {

  PetscCall(
      SlepcInitialize(&argc, &argv, (char *)0, "Calculate eigenvalues of A"));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Hello world!\n"));

  std::vector<PetscReal> omega_list{10.0, 20.0, 30.0, 40.0, 50.0,
                                    60.0, 70.0, 80.0, 90.0, 100.0};
  PetscInt pts_per_wavelen = 10;

  // Create or open hdf5 database.
  hid_t file_id = get_file(HDF5_DB);

  // Create or open root group.
  hid_t group_id = get_group(file_id, ROOT_GROUP);

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
    MatExpre<2> matexpre(interior_domain_lens, interior_elems, absorber_elems);

    // Create velocity vector.
    Vec velocity = nullptr;
    // Create matrix.
    Mat A = nullptr;
    // Create eigensolver.
    EPS eps = nullptr;
    // Number of converged eigenvalues.
    PetscInt nconv = 0;

    PetscCall(DMCreateGlobalVector(matexpre.dm, &velocity));
    PetscCall(VecSet(velocity, v_max));
    matexpre.velocity = velocity;

    PetscCall(DMCreateMatrix(matexpre.dm, &A));
    PetscCall(matexpre.get_mat(A));

    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, A, nullptr));
    PetscCall(EPSSetProblemType(eps, EPS_NHEP));
    PetscCall(EPSSetDimensions(eps, 8, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY));
    PetscCall(EPSSetFromOptions(eps));

    PetscCall(EPSSolve(eps));

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

    // Destroy objects.
    PetscCall(EPSDestroy(&eps));
    PetscCall(VecDestroy(&velocity));
    PetscCall(MatDestroy(&A));
    {
      // Write eigenvalues to hdf5 database.
      size_t len = eigvals.size();
      auto datasp_id = H5Screate_simple(1, &len, nullptr);

      auto complex_id = get_complex_dtype();

      auto dataset_name = "omega" + std::to_string(static_cast<int>(omega));
      // Create or open dataset.
      auto dataset_id =
          get_dataset(group_id, dataset_name, complex_id, datasp_id);

      add_attribute(group_id, dataset_name, "shape", "2d sq");
      add_attribute(group_id, dataset_name, "pts per", pts_per_wavelen);
      add_attribute(group_id, dataset_name, "abs rho", absorber_ratio);
      add_attribute(group_id, dataset_name, "omega", omega);
      add_attribute(group_id, dataset_name, "int dom len",
                    interior_domain_lens[0]);
      add_attribute(group_id, dataset_name, "int dom elm", interior_elems[0]);
      add_attribute(group_id, dataset_name, "abs elm", absorber_elems[0]);
      add_attribute(group_id, dataset_name, "v max", v_max);

      H5Dwrite(dataset_id, complex_id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               eigvals.data());

      H5Tclose(complex_id);
      H5Sclose(datasp_id);
    }
  }

  // Close hdf5 database.
  H5Gclose(group_id);
  H5Fclose(file_id);

  PetscCall(SlepcFinalize());
  return 0;
}