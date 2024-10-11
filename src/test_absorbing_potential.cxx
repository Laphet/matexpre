#include "matexpre.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewerhdf5.h"
#include <cmath>
#include <gsl/gsl_pow_int.h>
#include <gsl/gsl_specfunc.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

const PetscReal CENTER[3] = {0.5, 0.5, 0.5};
const PetscReal POINT_SOURCE_MAX = 128.0;
const std::string HDF5_FILENAME = "data.hdf5";
const std::string HDF5_GROUPNAME = "/test_absorbing_potential";

template <unsigned int DIM>
PetscErrorCode get_green_solution(MatExpre<DIM> &matexpre,
                                  const PetscReal pos[], Vec source,
                                  Vec solution, PetscInt point_source[]) {
  Vec vcoords = nullptr;
  DM cdm = nullptr, dm = matexpre.dm;
  void *acoords = nullptr, *asource = nullptr, *asolution = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xl = 0, yl = 0, zl = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xl, &yl, &zl));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &vcoords));
  PetscCall(DMDAVecGetArray(cdm, vcoords, &acoords));

  PetscCall(VecZeroEntries(source));
  PetscCall(VecZeroEntries(solution));
  PetscCall(DMDAVecGetArray(dm, source, &asource));
  PetscCall(DMDAVecGetArray(dm, solution, &asolution));

  if constexpr (DIM == 2) {
    // Get the point source index.
    point_source[0] =
        std::round(pos[0] / matexpre.h[0]) + matexpre.absorber_elems[0];
    point_source[1] =
        std::round(pos[1] / matexpre.h[1]) + matexpre.absorber_elems[1];

    DMDACoor2d **acoords_2d = reinterpret_cast<DMDACoor2d **>(acoords);
    PetscScalar **asource_2d = reinterpret_cast<PetscScalar **>(asource),
                **asolution_2d = reinterpret_cast<PetscScalar **>(asolution);

    for (PetscInt ey = ys; ey < ys + yl; ++ey)
      for (PetscInt ex = xs; ex < xs + xl; ++ex) {
        if (ex == point_source[0] && ey == point_source[1])
          asource_2d[ey][ex] = 1.0;

        PetscReal xy[] = {acoords_2d[ey][ex].x.real(),
                          acoords_2d[ey][ex].y.real()};
        PetscReal d_to_center =
            std::sqrt(gsl_pow_2(xy[0] - pos[0]) + gsl_pow_2(xy[1] - pos[1]));
        if (d_to_center > 0.0) {
          PetscScalar h0 = gsl_sf_bessel_J0(matexpre.omega * d_to_center) +
                           IU * gsl_sf_bessel_Y0(matexpre.omega * d_to_center);
          asolution_2d[ey][ex] = h0 * IU / 4.0;
        } else
          asolution_2d[ey][ex] = POINT_SOURCE_MAX;
      }
  }

  if constexpr (DIM == 3) {
    // Get the point source index.
    point_source[0] =
        std::round(pos[0] / matexpre.h[0]) + matexpre.absorber_elems[0];
    point_source[1] =
        std::round(pos[1] / matexpre.h[1]) + matexpre.absorber_elems[1];
    point_source[2] =
        std::round(pos[2] / matexpre.h[2]) + matexpre.absorber_elems[2];

    DMDACoor3d ***acoords_3d = reinterpret_cast<DMDACoor3d ***>(acoords);
    PetscScalar ***asource_3d = reinterpret_cast<PetscScalar ***>(asource),
                ***asolution_3d = reinterpret_cast<PetscScalar ***>(asolution);

    for (PetscInt ez = zs; ez < zs + zl; ++ez)
      for (PetscInt ey = ys; ey < ys + yl; ++ey)
        for (PetscInt ex = xs; ex < xs + xl; ++ex) {
          if (ex == point_source[0] && ey == point_source[1] &&
              ez == point_source[2])
            asource_3d[ez][ey][ex] = 1.0;

          PetscReal xyz[] = {acoords_3d[ez][ey][ex].x.real(),
                             acoords_3d[ez][ey][ex].y.real(),
                             acoords_3d[ez][ey][ex].z.real()};
          PetscReal d_to_center = std::sqrt(gsl_pow_2(xyz[0] - pos[0]) +
                                            gsl_pow_2(xyz[1] - pos[1]) +
                                            gsl_pow_2(xyz[2] - pos[2]));
          if (d_to_center > 0.0)
            asolution_3d[zs][ys][xs] =
                1.0 / 4.0 / PETSC_PI / d_to_center *
                std::exp(IU * matexpre.omega * d_to_center);
          else
            asolution_3d[zs][ys][xs] = POINT_SOURCE_MAX;
        }
  }

  PetscCall(DMDAVecRestoreArray(cdm, vcoords, &acoords));
  PetscCall(DMDAVecRestoreArray(dm, source, &asource));
  PetscCall(DMDAVecRestoreArray(dm, solution, &asolution));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode get_diff(MatExpre<DIM> &matexpre, const PetscInt pos[],
                        Vec num_solution, Vec green_solution, Vec diff) {
  Vec vcoords = nullptr;
  DM dm = matexpre.dm;
  void *anum_solution = nullptr, *agreen_solution = nullptr, *adiff = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xl = 0, yl = 0, zl = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xl, &yl, &zl));

  PetscCall(DMDAVecGetArray(dm, num_solution, &anum_solution));
  PetscCall(DMDAVecGetArray(dm, green_solution, &agreen_solution));
  PetscCall(DMDAVecGetArray(dm, diff, &adiff));

  if constexpr (DIM == 2) {
    PetscScalar **anum_solution_2d =
                    reinterpret_cast<PetscScalar **>(anum_solution),
                **agreen_solution_2d =
                    reinterpret_cast<PetscScalar **>(agreen_solution),
                **adiff_2d = reinterpret_cast<PetscScalar **>(adiff);

    for (PetscInt ey = ys; ey < ys + yl; ++ey)
      for (PetscInt ex = xs; ex < xs + xl; ++ex) {
        if (matexpre.absorber_elems[0] <= ex &&
            ex <= matexpre.absorber_elems[0] + matexpre.interior_elems[0] &&
            matexpre.absorber_elems[1] <= ey &&
            ey <= matexpre.absorber_elems[1] + matexpre.interior_elems[1]) {
          // Mannually set the infinity value in Green's function equal to the
          // numerical solution.
          if (ex == pos[0] && ey == pos[1])
            agreen_solution_2d[ey][ex] = anum_solution_2d[ey][ex];

          adiff_2d[ey][ex] =
              anum_solution_2d[ey][ex] - agreen_solution_2d[ey][ex];
        } else
          adiff_2d[ey][ex] = 0.0;
      }
  }

  if constexpr (DIM == 3) {
    PetscScalar ***anum_solution_3d =
                    reinterpret_cast<PetscScalar ***>(anum_solution),
                ***agreen_solution_3d =
                    reinterpret_cast<PetscScalar ***>(agreen_solution),
                ***adiff_3d = reinterpret_cast<PetscScalar ***>(adiff);

    for (PetscInt ez = zs; ez < zs + zl; ++ez)
      for (PetscInt ey = ys; ey < ys + yl; ++ey)
        for (PetscInt ex = xs; ex < xs + xl; ++ex) {
          if (matexpre.absorber_elems[0] <= ex &&
              ex <= matexpre.absorber_elems[0] + matexpre.interior_elems[0] &&
              matexpre.absorber_elems[1] <= ey &&
              ey <= matexpre.absorber_elems[1] + matexpre.interior_elems[1] &&
              matexpre.absorber_elems[2] <= ez &&
              ez <= matexpre.absorber_elems[2] + matexpre.interior_elems[2]) {
            // Mannually set the infinity value in Green's function equal to the
            // numerical solution.
            if (ex == pos[0] && ey == pos[1] && ez == pos[2])
              agreen_solution_3d[ez][ey][ex] = anum_solution_3d[ez][ey][ex];

            adiff_3d[ez][ey][ex] =
                anum_solution_3d[ez][ey][ex] - agreen_solution_3d[ez][ey][ex];
          } else
            adiff_3d[ez][ey][ex] = 0.0;
        }
  }

  PetscCall(DMDAVecRestoreArray(dm, num_solution, &anum_solution));
  PetscCall(DMDAVecRestoreArray(dm, green_solution, &agreen_solution));
  PetscCall(DMDAVecRestoreArray(dm, diff, &adiff));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode get_test_case(PetscInt interior_elem, PetscInt absorber_elem,
                             PetscReal omega, PetscReal eta,
                             const PetscReal pos[]) {
  std::vector<PetscReal> interior_domain_lens(DIM, 1.0);
  std::vector<PetscInt> interior_elems(DIM, interior_elem);
  std::vector<PetscInt> absorber_elems(DIM, absorber_elem);
  std::vector<PetscInt> point_source(DIM, -1);
  Vec velocity = nullptr, source = nullptr, green_solution = nullptr,
      num_solution = nullptr, diff = nullptr;
  Mat A = nullptr;
  KSP ksp = nullptr;
  PetscViewer viewer = nullptr;
  // Lambda function to append attributes to the dataset.
  auto append_attr = [&](PetscObject obj) {
    PetscInt dim = DIM;
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "dim", PETSC_INT, &dim);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "interior_elem", PETSC_INT,
                                        &interior_elem);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "absorber_elem", PETSC_INT,
                                        &absorber_elem);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "omega", PETSC_REAL,
                                        &omega);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "eta", PETSC_REAL, &eta);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "pos_x", PETSC_REAL,
                                        &pos[0]);
    PetscViewerHDF5WriteObjectAttribute(viewer, obj, "pos_y", PETSC_REAL,
                                        &pos[1]);
    if constexpr (DIM == 3)
      PetscViewerHDF5WriteObjectAttribute(viewer, obj, "pos_z", PETSC_REAL,
                                          &pos[2]);
  };
  // Set the name for the dataset.
  std::stringstream ss;
  ss << "i" << interior_elem << "_a" << absorber_elem;
  ss << std::fixed << std::setprecision(2) << "_o" << omega << "_e" << eta;
  std::string ds_basename = ss.str();

  PetscFunctionBeginUser;
  MatExpre<DIM> matexpre(interior_domain_lens.data(), interior_elems.data(),
                         absorber_elems.data());
  PetscCall(DMCreateGlobalVector(matexpre.dm, &velocity));
  PetscCall(VecSet(velocity, 1.0));
  matexpre.omega = omega;
  matexpre.eta = eta;
  matexpre.velocity = velocity;
  PetscCall(matexpre.print_info());

  // Get Green's function and delta source.
  PetscCall(DMCreateGlobalVector(matexpre.dm, &source));
  PetscCall(DMCreateGlobalVector(matexpre.dm, &green_solution));
  PetscCall(get_green_solution<DIM>(matexpre, pos, source, green_solution,
                                    point_source.data()));

  // Get the matrix.
  PetscCall(DMCreateMatrix(matexpre.dm, &A));
  PetscCall(matexpre.get_mat(A));

  // Solve the linear system.
  PetscCall(DMCreateGlobalVector(matexpre.dm, &num_solution));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, source, num_solution));

  // Compare the solution.
  PetscCall(VecDuplicate(num_solution, &diff));
  PetscCall(get_diff<DIM>(matexpre, point_source.data(), num_solution,
                          green_solution, diff));

  // Save the results.
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, HDF5_FILENAME.c_str(),
                                FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, HDF5_GROUPNAME.c_str()));
  // Save Green's function.
  std::string green_ds_name = ds_basename + "_g";
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(green_solution),
                               green_ds_name.c_str()));
  PetscCall(VecView(green_solution, viewer));
  append_attr(reinterpret_cast<PetscObject>(green_solution));
  // Save the numerical solution.
  std::string num_ds_name = ds_basename + "_n";
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(num_solution),
                               num_ds_name.c_str()));
  PetscCall(VecView(num_solution, viewer));
  append_attr(reinterpret_cast<PetscObject>(num_solution));
  // Save the difference.
  std::string diff_ds_name = ds_basename + "_d";
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(diff),
                               diff_ds_name.c_str()));
  PetscCall(VecView(diff, viewer));
  append_attr(reinterpret_cast<PetscObject>(diff));

  // Clean up.
  PetscCall(VecDestroy(&velocity));
  PetscCall(VecDestroy(&source));
  PetscCall(VecDestroy(&green_solution));
  PetscCall(VecDestroy(&num_solution));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&diff));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

  PetscInt interior_elem = 64, absorber_elem = 8;
  PetscReal omega = 10.0, eta = 25.0;
  PetscReal pos[] = {0.5, 0.5, 0.5};

  // Test the 2D case.
  PetscCall(get_test_case<2>(interior_elem, absorber_elem, omega, eta, pos));

  PetscCall(PetscFinalize());
  return 0;
}