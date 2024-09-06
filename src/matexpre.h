#pragma once
#include "petscdm.h"
#include "petscmat.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <complex>
#include <cstddef>

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

  PetscErrorCode _setup();

public:
  // To represent the computational domain, excluding periodic boundary
  // relations.
  DM dm;
  // Frequency.
  PetscReal omega;
  // Absorbing constant.
  PetscReal eta;

  MatExpre();

  MatExpre(const PetscReal _interior_domain_lens[DIM],
           const PetscInt _interior_elems[DIM],
           const PetscInt _absorber_elems[DIM], PetscReal _omega = 10.0,
           PetscReal _eta = 25.0);

  PetscErrorCode get_mat(Vec velocity, Mat A);

  PetscErrorCode print_info();

  ~MatExpre();
};
