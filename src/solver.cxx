#include "solver.h"
/***
Only be used for cap.
#include "gsl/gsl_errno.h"
#include "gsl/gsl_pow_int.h"
#include "gsl/gsl_sf_elljac.h"
***/
#include "petscdm.h"
#include "petscdmda.h"
#include "petscdmdatypes.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "petscviewerhdf5.h"
#include "slepcfn.h"
#include "slepcmfn.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

template <unsigned int DIM>
std::complex<double> Solver<DIM>::get_g(const double r, const double omega,
                                        const double c,
                                        const double absorber_len_neg,
                                        const double interior_domain_len,
                                        const double absorber_len_pos) {
  // Can we treat zero Dirichlet BC uniformly if PML is not used?
  if (r < 0.0 && absorber_len_neg > 0.0) {
    double temp = -r / absorber_len_neg;
    return 1.0 + IU * (c * temp * temp / absorber_len_neg) / omega;
  } else if (r > interior_domain_len && absorber_len_pos > 0.0) {
    double temp = (r - interior_domain_len) / absorber_len_pos;
    return 1.0 + IU * (c * temp * temp / absorber_len_pos) / omega;
  } else
    return std::complex<double>(1.0, 0.0);
}

// interior_elems/absorber_elems_neg/absorber_elems_pos/interior_domain_lens
// should be prepared.
template <unsigned int DIM> PetscErrorCode Solver<DIM>::_setup() {
  // Receive the CML arguments.
  pml_c = 20.0;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_c", &pml_c, nullptr));
  PetscCheck(pml_c >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "pml_c must be non-negative, but got %f.", pml_c);

  for (unsigned int i = 0; i < DIM; ++i) {
    total_dofs[i] =
        interior_elems[i] + absorber_elems_neg[i] + absorber_elems_pos[i] + 1;
    h[i] = interior_domain_lens[i] / interior_elems[i];
    absorber_lens_neg[i] = h[i] * absorber_elems_neg[i];
    absorber_lens_pos[i] = h[i] * absorber_elems_pos[i];
  }

  PetscFunctionBeginUser;

  if constexpr (DIM == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX, total_dofs[0], total_dofs[1],
                           PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr,
                           &dm));

    PetscCall(DMSetUp(dm));

    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens_neg[0],
        interior_domain_lens[0] + absorber_lens_pos[0], -absorber_lens_neg[1],
        interior_domain_lens[1] + absorber_lens_pos[1], 0.0, 0.0));
  }

  if constexpr (DIM == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, total_dofs[0],
                           total_dofs[1], total_dofs[2], PETSC_DECIDE,
                           PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr,
                           nullptr, &dm));
    PetscCall(DMSetUp(dm));

    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens_neg[0],
        interior_domain_lens[0] + absorber_lens_pos[0], -absorber_lens_neg[1],
        interior_domain_lens[1] + absorber_lens_pos[1], -absorber_lens_neg[2],
        interior_domain_lens[2] + absorber_lens_pos[2]));
  }

  // Get the coordinates.
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &vcoords));
  // Get DMDA information.
  PetscCall(
      DMDAGetCorners(dm, &x_start, &y_start, &z_start, &x_len, &y_len, &z_len));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM> PetscErrorCode Solver<DIM>::get_dm(DM *dm_out) {
  PetscFunctionBeginUser;
  *dm_out = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_vec_from_func(Vec v, func_ptr f, void *ctx) {
  // Stack variables.
  void *acoords = nullptr, *av = nullptr;

  PetscFunctionBeginUser;

  PetscCall(DMDAVecGetArray(cdm, vcoords, &acoords));
  PetscCall(DMDAVecGetArray(dm, v, &av));

  if constexpr (DIM == 2) {
    DMDACoor2d **acoords_2d = reinterpret_cast<DMDACoor2d **>(acoords);
    PetscScalar **av_2d = reinterpret_cast<PetscScalar **>(av);

    for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
        double x_coord = acoords_2d[y_ind][x_ind].x.real(),
               y_coord = acoords_2d[y_ind][x_ind].y.real();
        av_2d[y_ind][x_ind] = f(x_coord, y_coord, 0.0, ctx);
      }
  }

  if constexpr (DIM == 3) {
    DMDACoor3d ***acoords_3d = reinterpret_cast<DMDACoor3d ***>(acoords);
    PetscScalar ***av_3d = reinterpret_cast<PetscScalar ***>(av);

    for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
          double x_coord = acoords_3d[z_ind][y_ind][x_ind].x.real(),
                 y_coord = acoords_3d[z_ind][y_ind][x_ind].y.real(),
                 z_coord = acoords_3d[z_ind][y_ind][x_ind].z.real();
          av_3d[z_ind][y_ind][x_ind] = f(x_coord, y_coord, z_coord, ctx);
        }
  }

  PetscCall(DMDAVecRestoreArray(dm, v, &av));
  PetscCall(DMDAVecRestoreArray(cdm, vcoords, &acoords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_laplace_pml_mat(Mat mat, const double omega) {
  // Stack variables.
  void *acoords = nullptr;

  PetscFunctionBeginUser;

  PetscCall(DMDAVecGetArray(cdm, vcoords, &acoords));

  if constexpr (DIM == 2) {
    double hx = h[0], hy = h[1];
    DMDACoor2d **acoords_2d = reinterpret_cast<DMDACoor2d **>(acoords);
    for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
        double x = acoords_2d[y_ind][x_ind].x.real(),
               y = acoords_2d[y_ind][x_ind].y.real();
        double x_mhalf = x - 0.5 * hx, x_phalf = x + 0.5 * hx;
        double y_mhalf = y - 0.5 * hy, y_phalf = y + 0.5 * hy;
        std::complex<double> temp_a =
            1.0 / (hx * hx) /
            (get_g(x, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_lens_pos[0]) *
             get_g(x_mhalf, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_elems_pos[0]));
        std::complex<double> temp_b =
            1.0 / (hx * hx) /
            (get_g(x, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_lens_pos[0]) *
             get_g(x_phalf, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_lens_pos[0]));
        std::complex<double> temp_c =
            1.0 / (hy * hy) /
            (get_g(y, omega, pml_c, absorber_lens_neg[1],
                   interior_domain_lens[1], absorber_lens_pos[1]) *
             get_g(y_mhalf, omega, pml_c, absorber_lens_neg[1],
                   interior_domain_lens[1], absorber_lens_pos[1]));
        std::complex<double> temp_d =
            1.0 / (hy * hy) /
            (get_g(y, omega, pml_c, absorber_lens_neg[1],
                   interior_domain_lens[1], absorber_lens_pos[1]) *
             get_g(y_phalf, omega, pml_c, absorber_lens_neg[1],
                   interior_domain_lens[1], absorber_lens_pos[1]));

        MatStencil row = {0, y_ind, x_ind, 0};
        MatStencil cols[5] = {{0, y_ind, x_ind, 0},
                              {0, y_ind, x_ind - 1, 0},
                              {0, y_ind, x_ind + 1, 0},
                              {0, y_ind - 1, x_ind, 0},
                              {0, y_ind + 1, x_ind, 0}};
        PetscScalar vals[5] = {temp_a + temp_b + temp_c + temp_d, -temp_a,
                               -temp_b, -temp_c, -temp_d};
        // All diagonal elements are inserted, including boundary points.
        PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[0], &vals[0],
                                      INSERT_VALUES));
        // Points next to the boundary points need special treatments.
        if (x_ind - 1 >= 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[1], &vals[1],
                                        INSERT_VALUES));
        if (x_ind + 1 < total_dofs[0] - 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[2], &vals[2],
                                        INSERT_VALUES));
        if (y_ind - 1 >= 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[3], &vals[3],
                                        INSERT_VALUES));
        if (y_ind + 1 < total_dofs[1] - 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[4], &vals[4],
                                        INSERT_VALUES));
      }
  }

  if constexpr (DIM == 3) {
    double hx = h[0], hy = h[1], hz = h[2];
    DMDACoor3d ***acoords_3d = reinterpret_cast<DMDACoor3d ***>(acoords);
    for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
          double x = acoords_3d[z_ind][y_ind][x_ind].x.real(),
                 y = acoords_3d[z_ind][y_ind][x_ind].y.real(),
                 z = acoords_3d[z_ind][y_ind][x_ind].z.real();
          double x_mhalf = x - 0.5 * hx, x_phalf = x + 0.5 * hx;
          double y_mhalf = y - 0.5 * hy, y_phalf = y + 0.5 * hy;
          double z_mhalf = z - 0.5 * hz, z_phalf = z + 0.5 * hz;
          std::complex<double> temp_a =
              1.0 / (hx * hx) /
              (get_g(x, omega, pml_c, absorber_lens_neg[0],
                     interior_domain_lens[0], absorber_elems_pos[0]) *
               get_g(x_mhalf, omega, pml_c, absorber_lens_neg[0],
                     interior_domain_lens[0], absorber_elems_pos[0]));
          std::complex<double> temp_b =
              1.0 / (hx * hx) /
              (get_g(x, omega, pml_c, absorber_lens_neg[0],
                     interior_domain_lens[0], absorber_lens_pos[0]) *
               get_g(x_phalf, omega, pml_c, absorber_lens_neg[0],
                     interior_domain_lens[0], absorber_lens_pos[0]));
          std::complex<double> temp_c =
              1.0 / (hy * hy) /
              (get_g(y, omega, pml_c, absorber_lens_neg[1],
                     interior_domain_lens[1], absorber_lens_pos[1]) *
               get_g(y_mhalf, omega, pml_c, absorber_lens_neg[1],
                     interior_domain_lens[1], absorber_lens_pos[1]));
          std::complex<double> temp_d =
              1.0 / (hy * hy) /
              (get_g(y, omega, pml_c, absorber_lens_neg[1],
                     interior_domain_lens[1], absorber_lens_pos[1]) *
               get_g(y_phalf, omega, pml_c, absorber_lens_neg[1],
                     interior_domain_lens[1], absorber_lens_pos[1]));
          std::complex<double> temp_e =
              1.0 / (hz * hz) /
              (get_g(z, omega, pml_c, absorber_lens_neg[2],
                     interior_domain_lens[2], absorber_lens_pos[2]) *
               get_g(z_mhalf, omega, pml_c, absorber_lens_neg[2],
                     interior_domain_lens[2], absorber_lens_pos[2]));
          std::complex<double> temp_f =
              1.0 / (hz * hz) /
              (get_g(z, omega, pml_c, absorber_lens_neg[2],
                     interior_domain_lens[2], absorber_lens_pos[2]) *
               get_g(z_phalf, omega, pml_c, absorber_lens_neg[2],
                     interior_domain_lens[2], absorber_lens_pos[2]));

          MatStencil row = {z_ind, y_ind, x_ind, 0};
          MatStencil cols[7] = {
              {z_ind, y_ind, x_ind, 0},     {z_ind, y_ind, x_ind - 1, 0},
              {z_ind, y_ind, x_ind + 1, 0}, {z_ind, y_ind - 1, x_ind, 0},
              {z_ind, y_ind + 1, x_ind, 0}, {z_ind - 1, y_ind, x_ind, 0},
              {z_ind + 1, y_ind, x_ind, 0}};
          PetscScalar vals[7] = {temp_a + temp_b + temp_c + temp_d + temp_e +
                                     temp_f,
                                 -temp_a,
                                 -temp_b,
                                 -temp_c,
                                 -temp_d,
                                 -temp_e,
                                 -temp_f};

          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[0], &vals[0],
                                        INSERT_VALUES));
          if (x_ind - 1 >= 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[1], &vals[1],
                                          INSERT_VALUES));
          if (x_ind + 1 < total_dofs[0] - 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[2], &vals[2],
                                          INSERT_VALUES));
          if (y_ind - 1 >= 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[3], &vals[3],
                                          INSERT_VALUES));
          if (y_ind + 1 < total_dofs[1] - 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[4], &vals[4],
                                          INSERT_VALUES));
          if (z_ind - 1 >= 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[5], &vals[5],
                                          INSERT_VALUES));
          if (z_ind + 1 < total_dofs[2] - 1)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[6], &vals[6],
                                          INSERT_VALUES));
        }
  }

  // Assemble the matrix.
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cdm, vcoords, &acoords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_laplace_abc_mat(Mat mat, const double omega) {

  PetscFunctionBeginUser;

  if constexpr (DIM == 2) {
    double hx = h[0], hy = h[1];
    for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
        MatStencil row = {0, y_ind, x_ind, 0};
        // Interior points.
        if (1 <= x_ind && x_ind < total_dofs[0] - 1 && 1 <= y_ind &&
            y_ind < total_dofs[1] - 1) {
          MatStencil cols[5] = {{0, y_ind, x_ind, 0},
                                {0, y_ind, x_ind - 1, 0},
                                {0, y_ind, x_ind + 1, 0},
                                {0, y_ind - 1, x_ind, 0},
                                {0, y_ind + 1, x_ind, 0}};
          PetscScalar vals[5] = {2.0 / (hx * hx) + 2.0 / (hy * hy),
                                 -1.0 / (hx * hx), -1.0 / (hx * hx),
                                 -1.0 / (hy * hy), -1.0 / (hy * hy)};
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 5, cols, vals, INSERT_VALUES));
        }
        // Left boundary points.
        else if (x_ind == 0 && 1 <= y_ind && y_ind < total_dofs[1] - 1) {
          MatStencil cols[2] = {
              {0, y_ind, x_ind, 0},
              {0, y_ind, x_ind + 1, 0},
          };
          PetscScalar vals[2] = {
              // 1.0 / (hx * hx) - IU * (omega / hx) / (1 + 0.5 * IU * omega *
              // hx),
              1.0 / (hx * hx) - IU * (omega / hx) - 1.0 / (hx * hx),
              -1.0 / (hx * hx)};
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 2, cols, vals, INSERT_VALUES));
        }
        // Right boundary points.
        else if (x_ind == total_dofs[0] - 1 && 1 <= y_ind &&
                 y_ind < total_dofs[1] - 1) {
          MatStencil cols[2] = {
              {0, y_ind, x_ind, 0},
              {0, y_ind, x_ind - 1, 0},
          };
          PetscScalar vals[2] = {
              // 1.0 / (hx * hx) - IU * (omega / hx) / (1 + 0.5 * IU * omega *
              // hx),
              1.0 / (hx * hx) - IU * (omega / hx) - 1.0 / (hx * hx),
              -1.0 / (hx * hx)};
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 2, cols, vals, INSERT_VALUES));
        }
        // Bottom boundary points.
        else if (1 <= x_ind && x_ind < total_dofs[0] - 1 && y_ind == 0) {
          MatStencil cols[2] = {
              {0, y_ind, x_ind, 0},
              {0, y_ind + 1, x_ind, 0},
          };
          PetscScalar vals[2] = {
              // 1.0 / (hy * hy) - IU * (omega / hy) / (1 + 0.5 * IU * omega *
              // hy),
              1.0 / (hy * hy) - IU * (omega / hy) - 1.0 / (hy * hy),
              -1.0 / (hy * hy)};
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 2, cols, vals, INSERT_VALUES));
        }
        // Top boundary points.
        else if (1 <= x_ind && x_ind < total_dofs[0] - 1 &&
                 y_ind == total_dofs[1] - 1) {
          MatStencil cols[2] = {
              {0, y_ind, x_ind, 0},
              {0, y_ind - 1, x_ind, 0},
          };
          PetscScalar vals[2] = {// 1.0 / (hy * hy) - IU * (omega / hy) / (1 +
                                 // 0.5 * IU * omega * hy),
                                 1.0 / (hy * hy) - IU * (omega / hy),
                                 -1.0 / (hy * hy)};
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 2, cols, vals, INSERT_VALUES));
        }
        // Conner points
        else {
          MatStencil cols[1] = {
              {0, y_ind, x_ind, 0},
          };
          PetscScalar vals[1] = {
              1.0 / (hx * hx) + 1.0 / (hy * hy),
          };
          PetscCall(
              MatSetValuesStencil(mat, 1, &row, 1, cols, vals, INSERT_VALUES));
        }
      }
  }

  // Assemble the matrix.
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/***
Not so good...
template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_laplace_cap_mat(Mat mat, const double omega) {
  const double MAGIC_CONSTANT1_W = 2.5;
  const double MAGIC_CONSTANT2_W = 1.0 / std::sqrt(2.0);
  Vec Wxyz[DIM] = {nullptr}, W = nullptr;
  void *aWxyz[DIM] = {nullptr}, *acoords = nullptr;

  PetscFunctionBeginUser;
  // Construct the Wxyz vectors.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMGetGlobalVector(dm, &Wxyz[i]));
    PetscCall(VecZeroEntries(Wxyz[i]));
    PetscCall(DMDAVecGetArray(dm, Wxyz[i], &aWxyz[i]));
  }
  PetscCall(DMDAVecGetArray(cdm, vcoords, &acoords));

  if constexpr (DIM == 2) {
    DMDACoor2d **acoords_2d = reinterpret_cast<DMDACoor2d **>(acoords);
    PetscScalar **aWxyz_2d[2] = {nullptr, nullptr};
    for (unsigned int i = 0; i < 2; ++i) {
      aWxyz_2d[i] = reinterpret_cast<PetscScalar **>(aWxyz[i]);
    }

    for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
        double xy[] = {acoords_2d[y_ind][x_ind].x.real(),
                       acoords_2d[y_ind][x_ind].y.real()};
        for (unsigned int i = 0; i < DIM; ++i) {
          if (xy[i] < 0.0) {
            PetscReal cn = 0.0, u = 0.0, sn = 0.0, dn = 0.0;
            u = MAGIC_CONSTANT1_W * std::abs(xy[i]) / absorber_lens_neg[i] *
                MAGIC_CONSTANT2_W;
            auto gsl_status =
                gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, &sn, &cn, &dn);
            PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                        "Fail to performing cn(%.5e, %.5e)!\n", u,
                        MAGIC_CONSTANT2_W);
            aWxyz_2d[i][y_ind][x_ind] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
          }
          if (xy[i] > interior_domain_lens[i]) {
            PetscReal cn = 0.0, u = 0.0, sn = 0.0, dn = 0.0;
            u = MAGIC_CONSTANT1_W * std::abs(xy[i] - interior_domain_lens[i]) /
                absorber_lens_pos[i] * MAGIC_CONSTANT2_W;
            auto gsl_status =
                gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, &sn, &cn, &dn);
            PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                        "Fail to performing cn(%.5e, %.5e)!\n", u,
                        MAGIC_CONSTANT2_W);
            aWxyz_2d[i][y_ind][x_ind] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
          }
        }
      }
  }

  if constexpr (DIM == 3) {
    PetscPrintf(PETSC_COMM_WORLD, "Under construction!\n");
  }

  // Restore the arrays.
  PetscCall(DMDAVecRestoreArray(cdm, vcoords, &acoords));
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArray(dm, Wxyz[i], &aWxyz[i]));
  }

  // Compute Wxyz => 1 - Wxyz / max Wxyz.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscReal WxyzMax = 0.0;
    PetscCall(VecMax(Wxyz[i], nullptr, &WxyzMax));
    PetscCall(VecScale(Wxyz[i], -1.0 / WxyzMax));
    PetscCall(VecShift(Wxyz[i], 1.0));
  }
  PetscCall(DMGetGlobalVector(dm, &W));
  // Because we have zeroed the Wxyz vectors, we do not need to zero the W.
  PetscCall(VecCopy(Wxyz[0], W));
  for (unsigned int i = 1; i < DIM; ++i) {
    PetscCall(VecPointwiseMult(W, Wxyz[i], W));
  }
  PetscCall(VecScale(W, -1.0));
  PetscCall(VecShift(W, 1.0));
  // W -> -i omega eta W.
  PetscCall(VecScale(W, -IU * omega * pml_c));

  // Restore the global vectors.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMRestoreGlobalVector(dm, &Wxyz[i]));
  }

  // Construct the matrix.
  if constexpr (DIM == 2) {
    double hx = h[0], hy = h[1];
    for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind) {
        MatStencil row = {0, y_ind, x_ind, 0};
        MatStencil cols[5] = {{0, y_ind, x_ind, 0},
                              {0, y_ind, x_ind - 1, 0},
                              {0, y_ind, x_ind + 1, 0},
                              {0, y_ind - 1, x_ind, 0},
                              {0, y_ind + 1, x_ind, 0}};
        PetscScalar vals[5] = {2.0 / (hx * hx) + 2.0 / (hy * hy),
                               -1.0 / (hx * hx), -1.0 / (hx * hx),
                               -1.0 / (hy * hy), -1.0 / (hy * hy)};
        PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[0], &vals[0],
                                      INSERT_VALUES));
        if (x_ind - 1 >= 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[1], &vals[1],
                                        INSERT_VALUES));
        if (x_ind + 1 < total_dofs[0] - 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[2], &vals[2],
                                        INSERT_VALUES));
        if (y_ind - 1 >= 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[3], &vals[3],
                                        INSERT_VALUES));
        if (y_ind + 1 < total_dofs[1] - 1)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[4], &vals[4],
                                        INSERT_VALUES));
      }
  }

  if constexpr (DIM == 3) {
    PetscPrintf(PETSC_COMM_WORLD, "Under construction!\n");
  }

  // Assemble the matrix.
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  // Get the matrix -Delta - i omega eta W.
  PetscCall(MatDiagonalSet(mat, W, ADD_VALUES));

  // Clean up.
  PetscCall(DMRestoreGlobalVector(dm, &W));

  PetscFunctionReturn(PETSC_SUCCESS);
}
***/

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_delta_rhs(Vec rhs, PetscInt i0_offset,
                                          PetscInt j0_offset,
                                          PetscInt k0_offset) {
  void *arhs = nullptr;

  PetscFunctionBeginUser;
  // PetscCall(VecZeroEntries(rhs));
  PetscCall(DMDAVecGetArray(dm, rhs, &arhs));
  if constexpr (DIM == 2) {
    PetscScalar **arhs_2d = reinterpret_cast<PetscScalar **>(arhs);
    PetscInt i0 = absorber_elems_neg[0] + i0_offset,
             j0 = absorber_elems_neg[0] + j0_offset;

    if (x_start <= i0 && i0 < x_start + x_len && y_start <= j0 &&
        j0 < y_start + y_len) {
      arhs_2d[j0][i0] = 1.0;
    }
  }

  if constexpr (DIM == 3) {
    PetscScalar ***arhs_3d = reinterpret_cast<PetscScalar ***>(arhs);
    PetscInt i0 = absorber_elems_neg[0] + i0_offset,
             j0 = absorber_elems_neg[0] + j0_offset,
             k0 = absorber_elems_neg[0] + k0_offset;
    if (x_start <= i0 && i0 < x_start + x_len && y_start <= j0 &&
        j0 < y_start + y_len && z_start <= k0 && k0 < z_start + z_len) {
      arhs_3d[k0][j0][i0] = 1.0;
    }
  }

  PetscCall(DMDAVecRestoreArray(dm, rhs, &arhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM> Solver<DIM>::~Solver() {
  // According to the PETSc manual, it is prefered to use PetscCallAbort.
  PetscCallAbort(PETSC_COMM_SELF, DMDestroy(&dm));
}

template <unsigned int DIM>
Solver<DIM>::Solver(const int uniform_interior_elems,
                    const int uniform_absorber_elems) {
  for (unsigned int i = 0; i < DIM; ++i) {
    interior_domain_lens[i] = 1.0;
    interior_elems[i] = uniform_interior_elems;
    absorber_elems_neg[i] = uniform_absorber_elems;
    absorber_elems_pos[i] = uniform_absorber_elems;
  }
  PetscCallAbort(PETSC_COMM_SELF, _setup());
}

template <unsigned int DIM>
Solver<DIM>::Solver(const int levels, const double ratio) {
  for (unsigned int i = 0; i < DIM; ++i) {
    interior_domain_lens[i] = 1.0;
    interior_elems[i] = std::floor((1 << levels) * ratio / 2) * 2;
    absorber_elems_neg[i] = ((1 << levels) - interior_elems[i]) / 2;
    absorber_elems_pos[i] = ((1 << levels) - interior_elems[i]) / 2;
  }
  PetscCallAbort(PETSC_COMM_SELF, _setup());
}

template <unsigned int DIM>
Solver<DIM>::Solver(const int uniform_absorber_elems,
                    const int interior_elems[],
                    const double interior_domain_lens[]) {
  for (unsigned int i = 0; i < DIM; ++i) {
    this->interior_domain_lens[i] = interior_domain_lens[i];
    this->interior_elems[i] = interior_elems[i];
    this->absorber_elems_neg[i] = uniform_absorber_elems;
    this->absorber_elems_pos[i] = uniform_absorber_elems;
  }
  PetscCallAbort(PETSC_COMM_SELF, _setup());
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::read_hdf5_vec(Vec v, const char *hdf5_filename,
                                          const char *hdf5_groupname,
                                          const char *vec_int_name) {
  // Stack variables.
  DM dm_int = nullptr;
  Vec v_int = nullptr;
  PetscViewer hdf5_viewer = nullptr;
  IS is = nullptr;
  VecScatter scatter = nullptr;

  PetscFunctionBeginUser;
  // Create the external DM.
  if constexpr (DIM == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR, interior_elems[0] + 1,
                           interior_elems[1] + 1, PETSC_DECIDE, PETSC_DECIDE, 1,
                           1, nullptr, nullptr, &dm_int));
  }
  if constexpr (DIM == 3) {
    PetscCall(DMDACreate3d(
        PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR, interior_elems[0] + 1, interior_elems[1] + 1,
        interior_elems[2] + 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
        nullptr, nullptr, nullptr, &dm_int));
  }
  PetscCall(DMSetUp(dm_int));
  PetscCall(DMGetGlobalVector(dm_int, &v_int));
  PetscCall(
      PetscObjectSetName(reinterpret_cast<PetscObject>(v_int), vec_int_name));

  // Load the vector into v_int.
  std::string hdf5_full_filename =
      std::string(DATA_FOLDERPATH) + std::string(hdf5_filename);
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, hdf5_full_filename.c_str(),
                                FILE_MODE_READ, &hdf5_viewer));
  PetscCall(PetscViewerHDF5PushGroup(hdf5_viewer, hdf5_groupname));
  PetscCall(VecLoad(v_int, hdf5_viewer));

  // Copy the vector.
  PetscInt dm_int_x_start = 0, dm_int_y_start = 0, dm_int_z_start = 0,
           dm_int_x_len = 0, dm_int_y_len = 0, dm_int_z_len = 0;
  PetscCall(DMDAGetCorners(dm_int, &dm_int_x_start, &dm_int_y_start,
                           &dm_int_z_start, &dm_int_x_len, &dm_int_y_len,
                           &dm_int_z_len));
  // To large dm.
  MatStencil lower{0, dm_int_y_start + absorber_elems_neg[1],
                   dm_int_x_start + absorber_elems_neg[0], 0};
  MatStencil upper{0, dm_int_y_start + absorber_elems_neg[1] + dm_int_y_len,
                   dm_int_x_start + absorber_elems_neg[0] + dm_int_x_len, 0};
  if constexpr (DIM == 3) {
    lower.k = dm_int_z_start + absorber_elems_neg[2];
    upper.k = dm_int_z_start + absorber_elems_neg[2] + dm_int_z_len;
  }
  // Warning: Here should be PETSC_TRUE, otherwise the IS will be mismatched.
  PetscCall(DMDACreatePatchIS(dm, &lower, &upper, &is, PETSC_TRUE));

  PetscCall(VecScatterCreate(v_int, nullptr, v, is, &scatter));
  // Intialize v by one.
  PetscCall(VecSet(v, 1.0));
  PetscCall(VecScatterBegin(scatter, v_int, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, v_int, v, INSERT_VALUES, SCATTER_FORWARD));

  // Clean up.
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscViewerDestroy(&hdf5_viewer));
  PetscCall(DMRestoreGlobalVector(dm_int, &v_int));
  PetscCall(DMDestroy(&dm_int));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::print_info(const double omega) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Interior domain lengths: Lx=%.5f, Ly=%.5f",
                        interior_domain_lens[0], interior_domain_lens[1]));
  if constexpr (DIM == 3) {
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, ", Lz=%.5f.\n", interior_domain_lens[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Interior elements: Nx=%d, Ny=%d",
                        interior_elems[0], interior_elems[1]));
  if constexpr (DIM == 3) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", Nz=%d.\n", interior_elems[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Absorber elements: Nx=(%d, %d), Ny=(%d, %d)",
                        absorber_elems_neg[0], absorber_elems_pos[0],
                        absorber_elems_neg[1], absorber_elems_pos[1]));
  if constexpr (DIM == 3) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", Nz=(%d, %d).\n",
                          absorber_elems_neg[2], absorber_elems_pos[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Grid spacings: hx=%.5f, hy=%.5f",
                        h[0], h[1]));
  if constexpr (DIM == 3) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", hz=%.5f.\n", h[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PML constants: c=%.5f.\n", pml_c));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Omega: 2 Pi x %.5f=%.5f.\n",
                        omega / (2.0 * PETSC_PI), omega));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::save_xdmf_hdf5(Vec v,
                                           const char *xdmf_filename_surffix,
                                           const char *hdf5_filename,
                                           const char *hdf5_groupname) {
  // Stack variables.
  const char *vec_name = nullptr;
  std::string hdf5_full_filename =
      std::string(DATA_FOLDERPATH) + std::string(hdf5_filename);
  std::string xdmf_full_filename(DATA_FOLDERPATH);
  std::string xdmf_template_full_filename;
  if constexpr (DIM == 2) {
    xdmf_template_full_filename = std::string(XDMF_TEMPLATE2D_FILEPATH);
  } else {
    xdmf_template_full_filename = std::string(XDMF_TEMPLATE3D_FILEPATH);
  }
  std::string t_INFO("Target");
  std::string t_ADITIONAL_INFO;
  std::string t_GRID_DIMENSIONS;
  std::string t_GRID_ORIGIN;
  std::string t_GRID_SPACING;

  // Save the DMDA vector to .hdf5 file, the dimension of the hdf5 chunk will be
  // ny x nx x 2 in 2D, nz x ny x nx x 2 in 3D.
  for (int i = DIM - 1; i >= 0; --i) {
    t_ADITIONAL_INFO +=
        std::to_string(absorber_elems_neg[i]) + " " +
        std::to_string(absorber_elems_neg[i] + interior_elems[i] + 1);
    t_GRID_DIMENSIONS += std::to_string(total_dofs[i]);
    t_GRID_ORIGIN += std::to_string(static_cast<float>(-absorber_lens_neg[i]));
    t_GRID_SPACING += std::to_string(static_cast<float>(h[i]));
    if (i != 0) {
      t_ADITIONAL_INFO += " ";
      t_ADITIONAL_INFO += " ";
      t_GRID_DIMENSIONS += " ";
      t_GRID_ORIGIN += " ";
      t_GRID_SPACING += " ";
    }
  }

  PetscFunctionBeginUser;
  // Prepare names.
  PetscCall(PetscObjectGetName(reinterpret_cast<PetscObject>(v), &vec_name));
  xdmf_full_filename += std::string(vec_name) + "_" +
                        std::string(xdmf_filename_surffix) +
                        std::string(".xmf");
  // Save .hdf5 file.
  PetscViewer hdf5_viewer = nullptr;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, hdf5_full_filename.c_str(),
                                FILE_MODE_APPEND, &hdf5_viewer));
  PetscCall(PetscViewerHDF5PushGroup(hdf5_viewer, hdf5_groupname));
  PetscCall(VecView(v, hdf5_viewer));
  PetscCall(PetscViewerDestroy(&hdf5_viewer));

  // Read the template file.
  std::ifstream xdmf_template_file(xdmf_template_full_filename);
  // Open the new file.
  std::ofstream xdmf_file(xdmf_full_filename);

  // Write line by line.
  std::string line;
  // A lambda function to replace the string.
  auto line_replace = [](std::string &str, const std::string &from,
                         const std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
  };

  while (std::getline(xdmf_template_file, line)) {
    line_replace(line, "$INFO$", t_INFO);
    line_replace(line, "$ADITIONAL_INFO$", t_ADITIONAL_INFO);
    line_replace(line, "$GRID_DIMENSIONS$", t_GRID_DIMENSIONS);
    line_replace(line, "$ORIGIN$", t_GRID_ORIGIN);
    line_replace(line, "$SPACING$", t_GRID_SPACING);
    line_replace(line, "$HDF5_FILE$", hdf5_filename);
    line_replace(line, "$HDF5_GROUP$", hdf5_groupname);
    line_replace(line, "$VEC_NAME$", vec_name);

    xdmf_file << line << std::endl;
  }
  // xdmf_file.close();

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode Solver<DIM>::get_zeroed_boundary_vec(Vec v) {
  // Stack variables.
  void *av = nullptr;

  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArray(dm, v, &av));
  if constexpr (DIM == 2) {
    PetscScalar **av_2d = reinterpret_cast<PetscScalar **>(av);
    if (x_start == 0) {
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        av_2d[y_ind][0] = 0.0;
    }
    if (x_start + x_len == total_dofs[0]) {
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        av_2d[y_ind][total_dofs[0] - 1] = 0.0;
    }
    if (y_start == 0) {
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
        av_2d[0][x_ind] = 0.0;
    }
    if (y_start + y_len == total_dofs[1]) {
      for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
        av_2d[total_dofs[1] - 1][x_ind] = 0.0;
    }
  }

  if constexpr (DIM == 3) {
    PetscScalar ***av_3d = reinterpret_cast<PetscScalar ***>(av);
    if (x_start == 0) {
      for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
        for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
          av_3d[z_ind][y_ind][0] = 0.0;
    }
    if (x_start + x_len == total_dofs[0]) {
      for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
        for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
          av_3d[z_ind][y_ind][total_dofs[0] - 1] = 0.0;
    }
    if (y_start == 0) {
      for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
          av_3d[z_ind][0][x_ind] = 0.0;
    }
    if (y_start + y_len == total_dofs[1]) {
      for (PetscInt z_ind = z_start; z_ind < z_start + z_len; ++z_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
          av_3d[z_ind][total_dofs[1] - 1][x_ind] = 0.0;
    }
    if (z_start == 0) {
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
          av_3d[0][y_ind][x_ind] = 0.0;
    }
    if (z_start + z_len == total_dofs[2]) {
      for (PetscInt y_ind = y_start; y_ind < y_start + y_len; ++y_ind)
        for (PetscInt x_ind = x_start; x_ind < x_start + x_len; ++x_ind)
          av_3d[total_dofs[2] - 1][y_ind][x_ind] = 0.0;
    }
  }

  PetscCall(DMDAVecRestoreArray(dm, v, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Explicit instantiation.
template class Solver<2>;
template class Solver<3>;

PetscErrorCode get_shifted_velocity_mat(Mat A, Vec v, const PetscScalar alpha) {
  // Stack variables.
  DM dm = nullptr;
  Vec temp = nullptr;

  PetscFunctionBeginUser;
  // Get the DM from the vector.
  PetscCall(VecGetDM(v, &dm));
  // Create a temporary vector.
  PetscCall(DMGetGlobalVector(dm, &temp));
  // temp = alpha / v^2.
  PetscCall(VecPointwiseMult(temp, v, v));
  PetscCall(VecReciprocal(temp));
  PetscCall(VecScale(temp, alpha));
  // mat = mat + diag(temp).
  PetscCall(MatDiagonalSet(A, temp, ADD_VALUES));
  // Destroy the temporary vector.
  PetscCall(DMRestoreGlobalVector(dm, &temp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Functions.
std::complex<double> func_one(const double x, const double y, const double z,
                              void *ctx) {
  return 1.0;
}

std::complex<double> func_gaussian(const double x, const double y,
                                   const double z, void *_ctx) {
  auto ctx = reinterpret_cast<GaussianCtx *>(_ctx);
  double x0 = ctx->position[0], y0 = ctx->position[1], z0 = ctx->position[2];
  double d = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
  return ctx->coefficent * std::exp(-d / (2.0 * ctx->sigma * ctx->sigma));
}

// Complex shift preconditioner.
extern PetscErrorCode PCSetUp_ComplexShiftPre(PC pc) {
  // Stack variables.
  ComplexShiftPre *ctx = nullptr;
  Mat A = nullptr;
  PC P_pc = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-csp_shift", &ctx->shift,
                                nullptr));
  PetscCheck(ctx->shift >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The shift.im must be non-negative, but got %f.\n", ctx->shift);

  PetscCall(PCGetOperators(pc, &A, nullptr));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx->P_mat));
  PetscCall(get_shifted_velocity_mat(
      ctx->P_mat, ctx->velocity, ctx->omega * ctx->omega * ctx->shift * IU));
  // Set up the KSP for P^{-1} b = u.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ctx->P_ksp));
  PetscCall(KSPSetOperators(ctx->P_ksp, ctx->P_mat, ctx->P_mat));
  PetscCall(KSPGetPC(ctx->P_ksp, &P_pc));
  // Set default KSP type.
  PetscCall(KSPSetType(ctx->P_ksp, KSPBCGS));
  // Allow CML options.
  PetscCall(KSPSetOptionsPrefix(ctx->P_ksp, "csp_"));
  // KSPSetFromOptions will call PCSetFromOptions.
  // Now pc knows that its type is mg.
  PetscCall(KSPSetFromOptions(ctx->P_ksp));

  // Set the PCMG for the P_ksp.
  PetscBool use_pcmg = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(P_pc), PCMG,
                                   &use_pcmg));
  DM dm = nullptr;
  PetscCall(VecGetDM(ctx->velocity, &dm));
  if (use_pcmg) {
    PetscCall(PCMGSetupViaCoarsen(P_pc, dm));
  }
  // KSPSetUp setup will call PCSetUp.
  PetscCall(KSPSetUp(ctx->P_ksp));

#ifdef DEBUG
  // PetscCall(PCView(P_pc, PETSC_VIEWER_STDOUT_WORLD));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_ComplexShiftPre(PC pc, Vec in, Vec out) {
  // Stack variables.
  ComplexShiftPre *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(KSPSolve(ctx->P_ksp, in, out));

#ifdef DEBUG
  PetscCall(KSPConvergedReasonView(ctx->P_ksp, nullptr));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_ComplexShiftPre(PC pc) {
  // Stack variables.
  ComplexShiftPre *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(MatDestroy(&ctx->P_mat));
  // KSPDestroy will call PCDestroy.
  PetscCall(KSPDestroy(&ctx->P_ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCShell_ComplexShiftPre(PC pc, ComplexShiftPre *ctx) {
  PetscFunctionBeginUser;

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetApply(pc, PCApply_ComplexShiftPre));
  PetscCall(PCShellSetSetUp(pc, PCSetUp_ComplexShiftPre));
  PetscCall(PCShellSetDestroy(pc, PCDestroy_ComplexShiftPre));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// MatExPre.
extern PetscErrorCode PCSetUp_MatExPre(PC pc) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  Mat A = nullptr;
  FN fn = nullptr;

  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(PCGetOperators(pc, &A, nullptr));

  // Handle CML options.
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_steps", &ctx->steps,
                               nullptr));
  PetscCheck(ctx->steps > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The steps must be positive, but got %d.\n", ctx->steps);
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-matex_delta_t",
                                &ctx->delta_t, nullptr));
  PetscCheck(ctx->delta_t > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The delta_t must be positive, but got %f.\n", ctx->delta_t);

  // Create phi0.
  PetscCall(MFNCreate(PETSC_COMM_WORLD, &ctx->phi0));
  PetscCall(MFNSetOperator(ctx->phi0, A));
  PetscCall(MFNGetFN(ctx->phi0, &fn));
  PetscCall(FNSetType(fn, FNEXP));
  PetscCall(FNSetScale(fn, ctx->delta_t * IU, 1.0));
  PetscCall(MFNSetOptionsPrefix(ctx->phi0, "phi0_"));
  PetscCall(MFNSetFromOptions(ctx->phi0));
  PetscCall(MFNSetUp(ctx->phi0));

  // Create phi1.
  PetscCall(MFNCreate(PETSC_COMM_WORLD, &ctx->phi1));
  PetscCall(MFNSetOperator(ctx->phi1, A));
  PetscCall(MFNGetFN(ctx->phi1, &fn));
  PetscCall(FNSetType(fn, FNPHI));
  PetscCall(FNPhiSetIndex(fn, 1));
  PetscCall(FNSetScale(fn, ctx->delta_t * IU, 1.0));
  PetscCall(MFNSetOptionsPrefix(ctx->phi1, "phi1_"));
  PetscCall(MFNSetFromOptions(ctx->phi1));
  PetscCall(MFNSetUp(ctx->phi1));

  // Create phi2.
  PetscCall(MFNCreate(PETSC_COMM_WORLD, &ctx->phi2));
  PetscCall(MFNSetOperator(ctx->phi2, A));
  PetscCall(MFNGetFN(ctx->phi2, &fn));
  PetscCall(FNSetType(fn, FNPHI));
  PetscCall(FNPhiSetIndex(fn, 2));
  PetscCall(FNSetScale(fn, ctx->delta_t * IU, 1.0));
  PetscCall(MFNSetOptionsPrefix(ctx->phi2, "phi2_"));
  PetscCall(MFNSetFromOptions(ctx->phi2));
  PetscCall(MFNSetUp(ctx->phi2));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_MatExPre(PC pc, Vec in, Vec out) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  Vec E_in = nullptr, F_in = nullptr, phi_out = nullptr;

  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(VecDuplicate(in, &E_in));
  PetscCall(VecDuplicate(in, &F_in));
  PetscCall(VecDuplicate(in, &phi_out));

  // Intialization.
  PetscCall(VecCopy(in, E_in));
  PetscCall(VecZeroEntries(F_in));
  PetscCall(VecZeroEntries(out));

  // The main loop.
  for (auto k = 0; k < ctx->steps; ++k) {
    // phi2_out = phi2(i h A)E.
    PetscCall(MFNSolve(ctx->phi2, E_in, phi_out));
    // G^{k+1} = G^k + h F^k + h**2 phi2_out.
    PetscCall(VecAXPBYPCZ(out, ctx->delta_t, ctx->delta_t * ctx->delta_t, 1.0,
                          F_in, phi_out));

    // If we reach the last step, we do not need to update rest vectors.
    if (k + 1 == ctx->steps)
      break;

    // phi1_out = phi1(i h A)E.
    PetscCall(MFNSolve(ctx->phi1, E_in, phi_out));
    // F^{k+1} = F^k + h phi1_out.
    PetscCall(VecAXPY(F_in, ctx->delta_t, phi_out));

    // phi0_out = exp(i h A)E.
    PetscCall(MFNSolve(ctx->phi0, E_in, phi_out));
    // E^{k+1} <- phi0_out.
    PetscCall(VecCopy(phi_out, E_in));
  }
  PetscCall(VecScale(out, -IU / (ctx->delta_t * ctx->steps)));

  // Clean up.
  PetscCall(VecDestroy(&phi_out));
  PetscCall(VecDestroy(&F_in));
  PetscCall(VecDestroy(&E_in));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_MatExPre(PC pc) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(MFNDestroy(&ctx->phi0));
  PetscCall(MFNDestroy(&ctx->phi1));
  PetscCall(MFNDestroy(&ctx->phi2));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCShell_MatExPre(PC pc, MatExPre *ctx) {
  PetscFunctionBeginUser;

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetApply(pc, PCApply_MatExPre));
  PetscCall(PCShellSetSetUp(pc, PCSetUp_MatExPre));
  PetscCall(PCShellSetDestroy(pc, PCDestroy_MatExPre));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_finest) {
  // Stack variables.
  std::vector<DM> da_hierarchy;

  PetscFunctionBeginUser;
  PetscInt nlevels = 2;
  PetscCall(PCMGGetLevels(pc, &nlevels));
  PetscCheck(nlevels >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The number of levels must be at least 2, but got %d.\n", nlevels);

  // Finest is at 0.
  da_hierarchy.resize(nlevels);
  da_hierarchy[0] = da_finest;
  PetscCall(DMCoarsenHierarchy(da_hierarchy[0], nlevels - 1, &da_hierarchy[1]));

  PetscCall(PCMGSetLevels(pc, nlevels, nullptr));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetNumberSmooth(pc, 1));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));

  // Reverse the da_hierarchy, now the finest is at the end.
  std::reverse(da_hierarchy.begin(), da_hierarchy.end());

  for (auto k = 1; k < nlevels; ++k) {
    Mat R = nullptr;
    PetscCall(
        DMCreateInterpolation(da_hierarchy[k - 1], da_hierarchy[k], &R, NULL));
    PetscCall(PCMGSetInterpolation(pc, k, R));
    PetscCall(MatDestroy(&R));
  }

  // Do not destroy the finest level.
  for (auto k = 0; k < nlevels - 1; ++k) {
    PetscCall(DMDestroy(&da_hierarchy[k]));
  }

  // Define default solvers for each level.
  // It seems that CML options will supercede all things below.
  KSP ksp_each_level = nullptr;
  PC pc_each_level = nullptr;
  PetscCall(PCMGGetCoarseSolve(pc, &ksp_each_level));
  PetscCall(KSPSetType(ksp_each_level, KSPPREONLY));
  PetscCall(KSPGetPC(ksp_each_level, &pc_each_level));
  PetscCall(PCSetType(pc_each_level, PCLU));
  PetscCall(PCFactorSetMatSolverType(pc_each_level, MATSOLVERMKL_CPARDISO));
  for (auto k = 1; k < nlevels; ++k) {
    PetscCall(PCMGGetSmoother(pc, k, &ksp_each_level));
    PetscCall(KSPSetType(ksp_each_level, KSPBCGS));
    PetscCall(KSPGetPC(ksp_each_level, &pc_each_level));
    PetscCall(PCSetType(pc_each_level, PCBJACOBI));
  }

#ifdef DEBUG
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "PCMGSetupViaCoarsen is called with levels=%d.\n",
                        nlevels));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetUp_MatExPreVer2(PC pc) {
  // Stack variables.
  MatExPreVer2Ctx *ctx = nullptr;
  Mat A = nullptr;
  FN fn = nullptr;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  // Handle CML options.
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_steps", &ctx->steps,
                               nullptr));
  PetscCheck(ctx->steps > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The steps must be positive, but got %d.\n", ctx->steps);
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-matex_delta_t",
                                &ctx->delta_t, nullptr));
  PetscCheck(ctx->delta_t > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The delta_t must be positive, but got %f.\n", ctx->delta_t);
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-matex_shift", &ctx->shift,
                                nullptr));
  PetscCheck(ctx->shift >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The shift must be non-negative, but got %f.\n", ctx->shift);

  // Create the shifted matrix.
  PetscCall(PCGetOperators(pc, &A, nullptr));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx->P_mat));
  PetscCall(get_shifted_velocity_mat(
      ctx->P_mat, ctx->velocity, ctx->omega * ctx->omega * ctx->shift * IU));

  // Create phi0.
  PetscCall(MFNCreate(PETSC_COMM_WORLD, &ctx->phi0));
  PetscCall(MFNSetOperator(ctx->phi0, ctx->P_mat));
  PetscCall(MFNGetFN(ctx->phi0, &fn));
  PetscCall(FNSetType(fn, FNEXP));
  PetscCall(FNSetScale(fn, ctx->delta_t * IU, 1.0));
  PetscCall(MFNSetOptionsPrefix(ctx->phi0, "phi0_"));
  PetscCall(MFNSetFromOptions(ctx->phi0));
  PetscCall(MFNSetUp(ctx->phi0));

  // Create phi1.
  PetscCall(MFNCreate(PETSC_COMM_WORLD, &ctx->phi1));
  PetscCall(MFNSetOperator(ctx->phi1, ctx->P_mat));
  PetscCall(MFNGetFN(ctx->phi1, &fn));
  PetscCall(FNSetType(fn, FNPHI));
  PetscCall(FNPhiSetIndex(fn, 1));
  PetscCall(FNSetScale(fn, ctx->delta_t * IU, 1.0));
  PetscCall(MFNSetOptionsPrefix(ctx->phi1, "phi1_"));
  PetscCall(MFNSetFromOptions(ctx->phi1));
  PetscCall(MFNSetUp(ctx->phi1));

#ifdef DEBUG
  double expect_ratio = std::exp(-ctx->delta_t * ctx->shift * ctx->omega *
                                 ctx->omega * (ctx->steps + 1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "PCSetUp_MatExPreVer2 is called with delta_t=%.5e, "
                        "steps=%d, shift=%.5e, x=%.5e.\n",
                        ctx->delta_t, ctx->steps, ctx->shift, expect_ratio));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_MatExPreVer2(PC pc, Vec in, Vec out) {
  // Stack variables.
  MatExPreVer2Ctx *ctx = nullptr;
  Vec E_in = nullptr, phi_out = nullptr;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(VecDuplicate(in, &E_in));
  PetscCall(VecDuplicate(in, &phi_out));

  // phi_out = phi1(in).
  PetscCall(MFNSolve(ctx->phi1, in, phi_out));

  // Initialize out.
  // out <- phi_out.
  PetscCall(VecCopy(phi_out, out));
  // The main loop.
  for (auto k = 0; k < ctx->steps; ++k) {
    // E_in <- phi_out.
    PetscCall(VecCopy(phi_out, E_in));
    // phi_out = phi0(E_in).
    PetscCall(MFNSolve(ctx->phi0, E_in, phi_out));
    // out = out + phi_out.
    PetscCall(VecAXPY(out, 1.0, phi_out));
  }

  PetscCall(VecScale(out, -IU));

  // Clean up.
  PetscCall(VecDestroy(&phi_out));
  PetscCall(VecDestroy(&E_in));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_MatExPreVer2(PC pc) {
  // Stack variables.
  MatExPreVer2Ctx *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(MatDestroy(&ctx->P_mat));
  PetscCall(MFNDestroy(&ctx->phi0));
  PetscCall(MFNDestroy(&ctx->phi1));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCShell_MatExPreVer2(PC pc, MatExPreVer2Ctx *ctx) {
  PetscFunctionBeginUser;

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetApply(pc, PCApply_MatExPreVer2));
  PetscCall(PCShellSetSetUp(pc, PCSetUp_MatExPreVer2));
  PetscCall(PCShellSetDestroy(pc, PCDestroy_MatExPreVer2));

  PetscFunctionReturn(PETSC_SUCCESS);
}