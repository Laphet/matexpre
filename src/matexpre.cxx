#include "matexpre.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_pow_int.h"
#include "gsl/gsl_sf_elljac.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscdmtypes.h"
#include "petscerror.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <cstdlib>
#include <fftw3-mpi.h>
#include <gsl/gsl_pow_int.h>

template <unsigned int DIM> PetscErrorCode MatExpre<DIM>::_setup() {
  // Borrowed, do not destroy.
  Vec vcoords = nullptr;
  // Borrowed, do not destroy.
  DM cdm = nullptr;
  void *acoords = nullptr, *aWxyz[DIM] = {nullptr};
  // Use std::vector to store the global vectors.
  Vec Wxyz[DIM] = {nullptr};
  IS fftw_patch_is;

  PetscFunctionBeginUser;
  // Process other parameters.
  for (unsigned int i = 0; i < DIM; ++i) {
    total_elems[i] = interior_elems[i] + 2 * absorber_elems[i];
    h[i] = interior_domain_lens[i] / interior_elems[i];
    absorber_lens[i] = h[i] * absorber_elems[i];
  }

  // Create the DMDA for the physical domain, use c++17 if constexpr.
  if constexpr (DIM == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                           DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
                           total_elems[0], total_elems[1], PETSC_DECIDE,
                           PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm));

    PetscCall(DMSetUp(dm));
    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens[0], interior_domain_lens[0] + absorber_lens[0],
        -absorber_lens[1], interior_domain_lens[1] + absorber_lens[1], 0.0,
        0.0));
  }

  if constexpr (DIM == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                           DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                           DMDA_STENCIL_STAR, total_elems[0], total_elems[1],
                           total_elems[2], PETSC_DECIDE, PETSC_DECIDE,
                           PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm));
    PetscCall(DMSetUp(dm));

    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens[0], interior_domain_lens[0] + absorber_lens[0],
        -absorber_lens[1], interior_domain_lens[1] + absorber_lens[1],
        -absorber_lens[2], interior_domain_lens[2] + absorber_lens[2]));
  }

  // Create the W(r) vector.
  PetscCall(DMCreateGlobalVector(dm, &W));

  // Get the global vectors, which will be restored rather than destroyed.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMGetGlobalVector(dm, &Wxyz[i]));
    // Always zero the vectors from DMGetGlobalVector.
    PetscCall(VecZeroEntries(Wxyz[i]));
  }

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &vcoords));

  // Get the global vectors, Wx, Wy, whether Wz is needed depends on the DIM.
  for (unsigned int i = 0; i < DIM; ++i) {
    // Always zero the vectors from DMGetGlobalVector.
    PetscCall(VecZeroEntries(Wxyz[i]));
  }

  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecGetArray(dm, Wxyz[i], &aWxyz[i]));
  }
  PetscCall(DMDAVecGetArray(cdm, vcoords, &acoords));

  if constexpr (DIM == 2) {
    PetscInt xs = 0, ys = 0, xl = 0, yl = 0;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xl, &yl, nullptr));

    DMDACoor2d **acoords_2d = nullptr;
    PetscScalar **aWxyz_2d[2] = {nullptr, nullptr};

    for (unsigned int i = 0; i < 2; ++i) {
      aWxyz_2d[i] = reinterpret_cast<PetscScalar **>(aWxyz[i]);
    }
    acoords_2d = reinterpret_cast<DMDACoor2d **>(acoords);

    for (PetscInt yp = ys; yp < ys + yl; ++yp) {
      for (PetscInt xp = xs; xp < xs + xl; ++xp) {
        PetscReal xy[] = {acoords_2d[yp][xp].x.real(),
                          acoords_2d[yp][xp].y.real()};
        for (unsigned int i = 0; i < 2; ++i) {
          if (xy[i] < 0.0) {
            PetscReal cn = 0.0, u = 0.0, sn = 0.0, dn = 0.0;
            u = MAGIC_CONSTANT1_W * std::abs(xy[i]) / absorber_lens[i] *
                MAGIC_CONSTANT2_W;
            auto gsl_status =
                gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, &sn, &cn, &dn);
            PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                        "Fail to performing cn(%.5e, %.5e)", u,
                        MAGIC_CONSTANT2_W);
            aWxyz_2d[i][yp][xp] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
          }
          if (xy[i] > interior_domain_lens[i]) {
            PetscReal cn = 0.0, u = 0.0, sn = 0.0, dn = 0.0;
            u = MAGIC_CONSTANT1_W * std::abs(xy[i] - interior_domain_lens[i]) /
                absorber_lens[i] * MAGIC_CONSTANT2_W;
            auto gsl_status =
                gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, &sn, &cn, &dn);
            PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                        "Fail to performing cn(%.5e, %.5e)", u,
                        MAGIC_CONSTANT2_W);
            aWxyz_2d[i][yp][xp] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
          }
        }
      }
    }
  }

  if constexpr (DIM == 3) {
    PetscInt xs = 0, ys = 0, zs = 0, xl = 0, yl = 0, zl = 0;
    DMDACoor3d ***acoords_3d = nullptr;
    PetscScalar ***aWxyz_3d[3] = {nullptr, nullptr, nullptr};
    for (unsigned int i = 0; i < 3; ++i) {
      aWxyz_3d[i] = reinterpret_cast<PetscScalar ***>(aWxyz[i]);
    }
    acoords_3d = reinterpret_cast<DMDACoor3d ***>(acoords);

    for (PetscInt zp = zs; zp < zs + zl; ++zp) {
      for (PetscInt yp = ys; yp < ys + yl; ++yp) {
        for (PetscInt xp = xs; xp < xs + xl; ++xp) {
          PetscReal xyz[] = {acoords_3d[zp][yp][xp].x.real(),
                             acoords_3d[zp][yp][xp].y.real(),
                             acoords_3d[zp][yp][xp].z.real()};
          for (unsigned int i = 0; i < 3; ++i) {
            if (xyz[i] < 0.0) {
              PetscReal cn = 0.0, u = 0.0;
              u = MAGIC_CONSTANT1_W * std::abs(xyz[i]) / absorber_lens[i] *
                  MAGIC_CONSTANT2_W;
              auto gsl_status =
                  gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, nullptr, &cn, nullptr);
              PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                          "Fail to performing cn(%.5e, %.5e)", u,
                          MAGIC_CONSTANT2_W);
              aWxyz_3d[i][zp][yp][xp] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
            }
            if (xyz[i] > interior_domain_lens[i]) {
              PetscReal cn = 0.0, u = 0.0;
              u = MAGIC_CONSTANT1_W *
                  std::abs(xyz[i] - interior_domain_lens[i]) /
                  absorber_lens[i] * MAGIC_CONSTANT2_W;
              auto gsl_status =
                  gsl_sf_elljac_e(u, MAGIC_CONSTANT2_W, nullptr, &cn, nullptr);
              PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
                          "Fail to performing cn(%.5e, %.5e)", u,
                          MAGIC_CONSTANT2_W);
              aWxyz_3d[i][zp][yp][xp] = std::sqrt(1.0 / gsl_pow_4(cn) - 1.0);
            }
          }
        }
      }
    }
  }

  // Restore the arrays.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArray(dm, Wxyz[i], &aWxyz[i]));
  }
  PetscCall(DMDAVecRestoreArray(cdm, vcoords, &acoords));

  // Compute Wxyz => 1 - Wxyz / max Wxyz.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscReal WxyzMax = 0.0;
    PetscCall(VecMax(Wxyz[i], nullptr, &WxyzMax));
    PetscCall(VecScale(Wxyz[i], -1.0 / WxyzMax));
    PetscCall(VecShift(Wxyz[i], 1.0));
  }
  PetscCall(VecCopy(Wxyz[0], W));
  for (unsigned int i = 1; i < DIM; ++i) {
    PetscCall(VecPointwiseMult(W, Wxyz[i], W));
  }
  PetscCall(VecScale(W, -1.0));
  PetscCall(VecShift(W, 1.0));

  // Restore the global vectors.
  for (unsigned int i = 0; i < DIM; ++i) {
    PetscCall(DMRestoreGlobalVector(dm, &Wxyz[i]));
  }

  // FFTW3 context.
  fftw_mpi_init();
  // We want to follow PETSc's convention, and hence x is the fastest index.
  // Therefore, for FFTW3, we need to transpose the data.
  if constexpr (DIM == 2)
    alloc_local = fftw_mpi_local_size_2d_transposed(
        total_elems[1], total_elems[0], PETSC_COMM_WORLD, &local_n0,
        &local_0_start, &local_n1, &local_1_start);
  if constexpr (DIM == 3)
    alloc_local = fftw_mpi_local_size_3d_transposed(
        total_elems[2], total_elems[1], total_elems[0], PETSC_COMM_WORLD,
        &local_n0, &local_0_start, &local_n1, &local_1_start);
  fftw_data = fftw_alloc_complex(alloc_local);
  PetscAssert(fftw_data != nullptr, PETSC_COMM_WORLD,
              "Fail to allocate memory for fftw_data");
  if constexpr (DIM == 2) {
    forward_plan = fftw_mpi_plan_dft_2d(total_elems[1], total_elems[0],
                                        fftw_data, fftw_data, PETSC_COMM_WORLD,
                                        FFTW_FORWARD, FFTW_MPI_TRANSPOSED_OUT);
    backword_plan = fftw_mpi_plan_dft_2d(total_elems[1], total_elems[0],
                                         fftw_data, fftw_data, PETSC_COMM_WORLD,
                                         FFTW_BACKWARD, FFTW_MPI_TRANSPOSED_IN);
  }
  if constexpr (DIM == 3) {
    forward_plan = fftw_mpi_plan_dft_3d(
        total_elems[2], total_elems[1], total_elems[0], fftw_data, fftw_data,
        PETSC_COMM_WORLD, FFTW_FORWARD, FFTW_MPI_TRANSPOSED_OUT);
    backword_plan = fftw_mpi_plan_dft_3d(
        total_elems[2], total_elems[1], total_elems[0], fftw_data, fftw_data,
        PETSC_COMM_WORLD, FFTW_BACKWARD, FFTW_MPI_TRANSPOSED_IN);
  }

  // Warp the fftw_data into PETSc vector.
  PetscInt loc_vec_len = -1;
  if constexpr (DIM == 2) {
    loc_vec_len = local_n0 * total_elems[0];
  }
  if constexpr (DIM == 3) {
    loc_vec_len = local_n0 * total_elems[0] * total_elems[1];
  }

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, loc_vec_len,
                                  reinterpret_cast<PetscScalar *>(fftw_data),
                                  &fftw_vec_forward_input));

  MatStencil lower = {0, 0, 0, 0}, upper = {0, 0, total_elems[0], 0};
  if constexpr (DIM == 2) {
    lower.j = static_cast<PetscInt>(local_0_start);
    upper.j = static_cast<PetscInt>(local_0_start + local_n0);
  }
  if constexpr (DIM == 3) {
    lower.j = 0;
    upper.j = total_elems[1];
    lower.k = static_cast<PetscInt>(local_0_start);
    upper.k = static_cast<PetscInt>(local_0_start + local_n0);
  }
  PetscCall(DMDACreatePatchIS(dm, &lower, &upper, &fftw_patch_is, PETSC_TRUE));
  PetscCall(VecScatterCreate(W, fftw_patch_is, fftw_vec_forward_input, nullptr,
                             &scatter));
  PetscCall(ISDestroy(&fftw_patch_is));

  // Set other default values for public members.
  velocity = nullptr;
  omega = 10.0;
  eta = 25.0;
  tau = 1.0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM> MatExpre<DIM>::MatExpre() {
  // Initialize the default domain size.
  for (unsigned int i = 0; i < DIM; ++i) {
    interior_domain_lens[i] = 0.5;
    interior_elems[i] = 16;
    total_elems[i] = 32;
  }
  // Initialize the default frequency.
  omega = 10.0;
  eta = 25.0;

  PetscFunctionBeginUser;

  PetscCallVoid(
      PetscOptionsGetRealArray(nullptr, nullptr, "-interior-domain-lens",
                               &interior_domain_lens[0], nullptr, nullptr));
  PetscCallVoid(PetscOptionsGetIntArray(nullptr, nullptr, "-interior-elems",
                                        &interior_elems[0], nullptr, nullptr));
  PetscCallVoid(PetscOptionsGetIntArray(nullptr, nullptr, "-absorber-elems",
                                        &absorber_elems[0], nullptr, nullptr));
  PetscCallVoid(
      PetscOptionsGetReal(nullptr, nullptr, "-omega", &omega, nullptr));
  PetscCallVoid(PetscOptionsGetReal(nullptr, nullptr, "-eta", &eta, nullptr));

  PetscCallVoid(_setup());

  PetscFunctionReturnVoid();
}

template <unsigned int DIM>
MatExpre<DIM>::MatExpre(const PetscReal _interior_domain_lens[DIM],
                        const PetscInt _interior_elems[DIM],
                        const PetscInt _absorber_elems[DIM]) {
  for (unsigned int i = 0; i < DIM; ++i) {
    interior_domain_lens[i] = _interior_domain_lens[i];
    interior_elems[i] = _interior_elems[i];
    absorber_elems[i] = _absorber_elems[i];
  }

  PetscFunctionBeginUser;

  PetscCallVoid(_setup());

  PetscFunctionReturnVoid();
}

template <unsigned int DIM> PetscErrorCode MatExpre<DIM>::get_mat(Mat A) {
  // Recommended by PETSc, stack variables should be declared at the beginning.
  void *aW = nullptr, *avelocity = nullptr;

  PetscFunctionBeginUser;

  PetscAssert(
      velocity != nullptr, PETSC_COMM_WORLD,
      "Velocity field is not set, please set it before calling get_mat");

  // Get the arrays.
  PetscCall(DMDAVecGetArray(dm, W, &aW));
  PetscCall(DMDAVecGetArray(dm, velocity, &avelocity));

  if constexpr (DIM == 2) {
    PetscInt xs = 0, ys = 0, xl = 0, yl = 0;
    auto aW_2d = reinterpret_cast<PetscScalar **>(aW);
    auto avelocity_2d = reinterpret_cast<PetscScalar **>(avelocity);
    PetscReal hx = h[0], hy = h[1];

    PetscCall(DMDAGetCorners(dm, &xs, &ys, nullptr, &xl, &yl, nullptr));
    for (PetscInt yp = ys; yp < ys + yl; ++yp) {
      for (PetscInt xp = xs; xp < xs + xl; ++xp) {
        PetscReal v = avelocity_2d[yp][xp].real(), w = aW_2d[yp][xp].real();
        MatStencil row = {0, yp, xp, 0};
        MatStencil cols[MAX_STENCIL_2D] = {
            {0, yp, xp, 0},
            {0, yp, xp - 1 == -1 ? total_elems[0] - 1 : xp - 1, 0},
            {0, yp, xp + 1 == total_elems[0] ? 0 : xp + 1, 0},
            {0, yp - 1 == -1 ? total_elems[1] - 1 : yp - 1, xp, 0},
            {0, yp + 1 == total_elems[1] ? 0 : yp + 1, xp, 0}};
        // Laplacian operator.
        PetscScalar vals[MAX_STENCIL_2D] = {2.0 / hx / hx + 2.0 / hy / hy,
                                            -1.0 / hx / hx, -1.0 / hx / hx,
                                            -1.0 / hy / hy, -1.0 / hy / hy};
        vals[0] -= omega * omega / v / v;
        vals[0] -= eta * omega * w * IU;
        PetscCall(MatSetValuesStencil(A, 1, &row, MAX_STENCIL_2D, cols, vals,
                                      INSERT_VALUES));
      }
    }
  }

  if constexpr (DIM == 3) {
    PetscInt xs = 0, ys = 0, zs = 0, xl = 0, yl = 0, zl = 0;
    auto aW_3d = reinterpret_cast<PetscScalar ***>(aW);
    auto avelocity_3d = reinterpret_cast<PetscScalar ***>(avelocity);
    PetscReal hx = h[0], hy = h[1], hz = h[2];

    PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xl, &yl, &zl));
    for (PetscInt zp = zs; zp < zs + zl; ++zp) {
      for (PetscInt yp = ys; yp < ys + yl; ++yp) {
        for (PetscInt xp = xs; xp < xs + xl; ++xp) {
          PetscReal v = avelocity_3d[zp][yp][xp].real(),
                    w = aW_3d[zp][yp][xp].real();
          MatStencil row = {zp, yp, xp, 0};
          MatStencil cols[MAX_STENCIL_3D] = {
              {zp, yp, xp, 0},
              {zp, yp, xp - 1 == -1 ? total_elems[0] - 1 : xp - 1, 0},
              {zp, yp, xp + 1 == total_elems[0] ? 0 : xp + 1, 0},
              {zp, yp - 1 == -1 ? total_elems[1] - 1 : yp - 1, xp, 0},
              {zp, yp + 1 == total_elems[1] ? 0 : yp + 1, xp, 0},
              {zp - 1 == -1 ? total_elems[2] - 1 : zp - 1, yp, xp, 0},
              {zp + 1 == total_elems[2] ? 0 : zp + 1, yp, xp, 0}};
          // Laplacian operator.
          PetscScalar vals[MAX_STENCIL_3D] = {2.0 / hx / hx + 2.0 / hy / hy +
                                                  2.0 / hz / hz,
                                              -1.0 / hx / hx,
                                              -1.0 / hx / hx,
                                              -1.0 / hy / hy,
                                              -1.0 / hy / hy,
                                              -1.0 / hz / hz,
                                              -1.0 / hz / hz};
          vals[0] -= omega * omega / v / v;
          vals[0] -= eta * omega * w * IU;
          PetscCall(MatSetValuesStencil(A, 1, &row, MAX_STENCIL_3D, cols, vals,
                                        INSERT_VALUES));
        }
      }
    }
  }

  // Assemble the matrix.
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Restore the arrays.
  PetscCall(DMDAVecRestoreArray(dm, W, &aW));
  PetscCall(DMDAVecRestoreArray(dm, velocity, &avelocity));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM> PetscErrorCode MatExpre<DIM>::print_info() {
  PetscFunctionBeginUser;

  PetscPrintf(PETSC_COMM_WORLD, "Interior domain lengths: Lx=%.5f, Ly=%.5f",
              interior_domain_lens[0], interior_domain_lens[1]);
  if constexpr (DIM == 3) {
    PetscPrintf(PETSC_COMM_WORLD, ", Lz=%.5f\n", interior_domain_lens[2]);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }

  PetscPrintf(PETSC_COMM_WORLD, "Interior elements: Nx=%d, Ny=%d",
              interior_elems[0], interior_elems[1]);
  if constexpr (DIM == 3) {
    PetscPrintf(PETSC_COMM_WORLD, ", Nz=%d\n", interior_elems[2]);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }

  PetscPrintf(PETSC_COMM_WORLD, "Absorber elements: Nx=%d, Ny=%d",
              absorber_elems[0], absorber_elems[1]);
  if constexpr (DIM == 3) {
    PetscPrintf(PETSC_COMM_WORLD, ", Nz=%d\n", absorber_elems[2]);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }

  PetscPrintf(PETSC_COMM_WORLD, "Frequency: omega=%.5f\n", omega);
  PetscPrintf(PETSC_COMM_WORLD, "Absorbing constant: eta=%.5f\n", eta);

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM> MatExpre<DIM>::~MatExpre() {
  PetscFunctionBeginUser;

  PetscCallVoid(VecDestroy(&W));
  PetscCallVoid(DMDestroy(&dm));

  // Clean FFTW3 context.
  PetscCallVoid(VecDestroy(&fftw_vec_forward_input));
  PetscCallVoid(VecScatterDestroy(&scatter));
  fftw_destroy_plan(forward_plan);
  fftw_destroy_plan(backword_plan);
  fftw_free(fftw_data);
  fftw_mpi_cleanup();

  PetscFunctionReturnVoid();
}

template <unsigned int DIM>
PetscErrorCode MatExpre<DIM>::_DMDA_vec_to_FFTW_vec(Vec dmda_vec) {
  PetscFunctionBeginUser;

  PetscCall(VecScatterBegin(scatter, dmda_vec, fftw_vec_forward_input,
                            INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, dmda_vec, fftw_vec_forward_input,
                          INSERT_VALUES, SCATTER_FORWARD));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
PetscErrorCode MatExpre<DIM>::_FFTW_vec_to_DMDA_vec(Vec dmda_vec) {
  PetscFunctionBeginUser;

  PetscCall(VecScatterBegin(scatter, fftw_vec_forward_input, dmda_vec,
                            INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter, fftw_vec_forward_input, dmda_vec,
                          INSERT_VALUES, SCATTER_REVERSE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <unsigned int DIM>
void MatExpre<DIM>::_apply_exp_laplace_freq(PetscScalar coeff) {
  // After FFTW forward and with transposed output, locally, the data has the
  // shape of [local_n1][dim-y] in 2D and [local_n1][dim-z][dim-x] in 3D.

  if constexpr (DIM == 2) {
    for (auto fx = local_1_start; fx < local_1_start + local_n1; ++fx)
      for (auto fy = 0; fy < total_elems[1]; ++fy) {
        PetscReal l = 0.0;
        l += 4.0 * gsl_pow_2(std::sin(PETSC_PI * fx * h[0])) * total_elems[0] *
             total_elems[0];
        l += 4.0 * gsl_pow_2(std::sin(PETSC_PI * fy * h[1])) * total_elems[1] *
             total_elems[1];
        auto offset = (fx - local_1_start) * total_elems[1] + fy;
        PetscScalar *data_ptr =
            reinterpret_cast<PetscScalar *>(&fftw_data[offset]);
        *data_ptr *= std::exp(coeff * l);
      }
  }

  if constexpr (DIM == 3) {
    for (auto fy = local_1_start; fy < local_1_start + local_n1; ++fy)
      for (auto fz = 0; fz < total_elems[2]; ++fz)
        for (auto fx = 0; fx < total_elems[0]; ++fx) {
          PetscReal l = 0.0;
          l += 4.0 * gsl_pow_2(std::sin(PETSC_PI * fx * h[0])) *
               total_elems[0] * total_elems[0];
          l += 4.0 * gsl_pow_2(std::sin(PETSC_PI * fy * h[1])) *
               total_elems[1] * total_elems[1];
          l += 4.0 * gsl_pow_2(std::sin(PETSC_PI * fz * h[2])) *
               total_elems[2] * total_elems[2];
          auto offset = (fy - local_1_start) * total_elems[2] * total_elems[0] +
                        fz * total_elems[0] + fx;
          PetscScalar *data_ptr =
              reinterpret_cast<PetscScalar *>(&fftw_data[offset]);
          *data_ptr *= std::exp(coeff * l);
        }
  }
}

// Template instance for 2D.
template class MatExpre<2>;

// Template instance for 3D.
template class MatExpre<3>;