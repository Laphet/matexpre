#pragma once
#include "petscdm.h"
#include "petscpctypes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <complex>
#include <cstddef>
#include <fftw3-mpi.h>
#include <petscmat.h>

constexpr size_t MAX_DIM = 3;
constexpr size_t MAX_STENCIL_2D = 5;
constexpr size_t MAX_STENCIL_3D = 7;

const PetscReal MAGIC_CONSTANT1_W = 2.5;
const PetscReal MAGIC_CONSTANT2_W = 1.0 / std::sqrt(2.0);
const std::complex<double> IU = std::complex<double>(0.0, 1.0);

const PetscReal DEFAULT_INTERIOR_DOMAIN_LEN = 1.0;
const PetscInt DEFAULT_INTERIOR_ELEM = 16;
const PetscInt DEFAULT_ABSORBER_ELEM = 4;
const PetscReal DEFAULT_OMEGA = 8.0;
const PetscReal DEFAULT_ETA = 25.0;
const PetscReal DEFAULT_TAU = 1.0;
const PetscInt DEFAULT_QUAD_POINTS = 3;

const PetscReal GL_QUAD_POS_3[3] = {-1.0, 0.0, 1.0};
const PetscReal GL_QUAD_WGH_3[3] = {1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0};
const PetscReal GL_QUAD_POS_4[4] = {-1.0, -1.0 / std::sqrt(5.0),
                                    1.0 / std::sqrt(5.0), 1.0};
const PetscReal GL_QUAD_WGH_4[4] = {1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0};
const PetscReal GL_QUAD_POS_5[5] = {-1.0, -std::sqrt(3.0 / 7.0), 0.0,
                                    std::sqrt(3.0 / 7.0), 1.0};
const PetscReal GL_QUAD_WGH_5[5] = {1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0,
                                    49.0 / 90.0, 1.0 / 10.0};

template <unsigned int DIM> class MatExpre {
private:
  // The size of the domain in each direction (0, Lx)x(0, Ly)x(0, Lz).
  PetscReal interior_domain_lens[DIM];
  // The size of the domain including absorbing layers in each direction,
  // this should be computed from domainExt_size.
  PetscInt absorber_elems[DIM];
  // Number of cells in each direction.
  PetscInt interior_elems[DIM];
  // Number of cells including absorbing layers in each direction.
  PetscInt total_elems[DIM];
  // Cell sizes in each direction.
  PetscReal h[DIM];
  // The width of the absorbing layer in each direction.
  PetscReal absorber_lens[DIM];
  // The Petsc vector of W(r)
  Vec W;

  // FFTW context, just do not trust the PETSC FFTW interface.
  fftw_plan forward_plan, backward_plan;
  // The FFTW3 data array, the size is determined by FFTW, which may contain
  // more space than the local data.
  fftw_complex *fftw_data;
  // Use FFTW3 transpose functionality to improve the performance.
  ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;

  // The domain partition is determined by FFTW.
  Vec fftw_vec_forward_input;

  PetscErrorCode _setup();

  PetscErrorCode _DMDA_vec_to_FFTW_vec(Vec dmda_vec);

  PetscErrorCode _FFTW_vec_to_DMDA_vec(Vec dmda_vec);

  void _apply_exp_laplace_freq(PetscScalar coeff);

  PetscErrorCode _apply_exp_diag(PetscScalar coeff);

  PetscErrorCode _apply_exp_A(PetscScalar coeff);

public:
  DM dm;
  // The Petsc vector of the velocity field, initialized by the user.
  // Users should set the velocity field before calling the apply method.
  // Users should destroy the vector.
  Vec velocity;
  // Frequency.
  PetscReal omega;
  // Absorbing constant.
  PetscReal eta;
  // The exp integration time length.
  PetscReal tau;
  // The number of quandrature points in [0, tau].
  PetscInt num_quad_points;

  MatExpre();

  MatExpre(const PetscReal _interior_domain_lens[DIM],
           const PetscInt _interior_elems[DIM],
           const PetscInt _absorber_elems[DIM]);

  PetscErrorCode get_mat(Mat A);

  PetscErrorCode print_info();

  PetscErrorCode pc_apply(Vec input, Vec output);

  ~MatExpre();
};

// Warp the pc_apply method into a C interface.
extern "C" PetscErrorCode pc_apply_2d(PC pc, Vec input, Vec output);
extern "C" PetscErrorCode pc_apply_3d(PC pc, Vec input, Vec output);
