#pragma once
#include "petscdm.h"
#include "petscmat.h"
#include "petscsftypes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <complex>
#include <cstddef>
#include <fftw3-mpi.h>

constexpr size_t MAX_DIM = 3;
constexpr size_t MAX_STENCIL_2D = 5;
constexpr size_t MAX_STENCIL_3D = 7;
const PetscReal MAGIC_CONSTANT1_W = 2.5;
const PetscReal MAGIC_CONSTANT2_W = 1.0 / std::sqrt(2.0);
const std::complex<double> IU = std::complex<double>(0.0, 1.0);

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
  fftw_plan forward_plan, backword_plan;
  fftw_complex *fftw_data;
  // Use FFTW3 transpose functionality to improve the performance.
  ptrdiff_t alloc_local, local_n0, local_0_start, local_n1, local_1_start;

  // Wrap FFTW vector into PETSc vector for data transfer.
  // This is a seq vector.
  Vec fftw_vec_forward_input;
  VecScatter scatter;

  PetscErrorCode _setup();

  PetscErrorCode _DMDA_vec_to_FFTW_vec(Vec dmda_vec);

  PetscErrorCode _FFTW_vec_to_DMDA_vec(Vec dmda_vec);

  void _apply_exp_laplace_freq(PetscScalar coeff);

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

  MatExpre();

  MatExpre(const PetscReal _interior_domain_lens[DIM],
           const PetscInt _interior_elems[DIM],
           const PetscInt _absorber_elems[DIM]);

  PetscErrorCode get_mat(Mat A);

  PetscErrorCode print_info();

  PetscErrorCode pc_apply(Vec x, Vec y);

  ~MatExpre();
};
