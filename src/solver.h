#include "petscdm.h"
#include <complex>
#include <cstddef>

// File paths, do not know how to avoid macros.
// Assume the WD is bin/.
#define XDMF_TEMPLATE2D_FILEPATH "../config/2d_template.xmf"
#define XDMF_TEMPLATE3D_FILEPATH "../config/3d_template.xmf"
#define DATA_FOLDERPATH "../data/"

constexpr size_t MAX_DIM = 3;

const std::complex<double> IU = std::complex<double>(0.0, 1.0);

using func_ptr = std::complex<double> (*)(const double, const double,
                                          const double, void *);

template <unsigned int DIM> class Solver {
private:
  // The size of the domain in each direction (0, Lx)x(0, Ly)x(0, Lz).
  PetscReal interior_domain_lens[DIM];
  // The Number of cells in the interior domain, which we care about.
  PetscInt interior_elems[DIM];
  // The number of cells of in absorbing layers in each direction,
  PetscInt absorber_elems[DIM];

  // total_dof = interior_elems + 2*absorber_elems - 1.
  PetscInt total_dofs[DIM];
  // The cell sizes in each direction.
  PetscReal h[DIM];
  // The width of the absorbing layer in each direction.
  PetscReal absorber_lens[DIM];

  // dz = (1 + i sigma(x) / omega) dx.
  // sigma(x) = c (ratio)^2 / absorber_len.
  PetscReal pml_c[DIM];

  // Coordinate DMDA.
  DM cdm;
  Vec vcoords;

  // DM information.
  PetscInt x_start, y_start, z_start;
  PetscInt x_len, y_len, z_len;

  static std::complex<double> get_g(const double r, const double omega,
                                    const double c, const double absorber_len,
                                    const double interior_domain_len);

  PetscErrorCode _setup();

public:
  // The main DM object.
  DM dm;

  PetscErrorCode get_vec_from_func(Vec v, func_ptr f, void *ctx);

  PetscErrorCode get_laplace_mat(Mat A, const double omega);

  PetscErrorCode get_final_mat(Mat A, Vec v, const double omega);

  PetscErrorCode print_info();

  // The .hdf5 and .xmf files will be save in the DATA_FOLDERPATH.
  // The dataset of the Petsc vector will be saved in the location
  // hdf5_groupname/Vec_name. The .xmf filename will be Vec_name +
  // xdmf_filename_surffix.
  PetscErrorCode save_xdmf_hdf5(Vec v, const char *xdmf_filename_surffix,
                                const char *hdf5_filename,
                                const char *hdf5_groupname);

  Solver(const int uniform_interior_elems, const int uniform_absorber_elems);

  ~Solver();
};

// Functions.
std::complex<double> func_one(const double x, const double y, const double z,
                              void *ctx);

struct GaussianCtx {
  double position[MAX_DIM];
  double sigma;
  std::complex<double> coefficent;
};

std::complex<double> func_gaussian(const double x, const double y,
                                   const double z, void *ctx);
