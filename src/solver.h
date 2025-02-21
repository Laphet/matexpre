#include "petscksp.h"
#include "slepcmfn.h"
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
  double interior_domain_lens[DIM];
  // The Number of cells in the interior domain, which we care about.
  PetscInt interior_elems[DIM];
  // The number of cells of in absorbing layers in each negative direction.
  PetscInt absorber_elems_neg[DIM];
  // The number of cells of in absorbing layers in each positive direction.
  PetscInt absorber_elems_pos[DIM];

  // The total physical points in each direction.
  // We treat Dirichlet boundary conditions as a dof.
  // total_dof = interior_elems + absorber_elems_neg + absorber_elems_pos + 1.
  PetscInt total_dofs[DIM];
  // The cell sizes in each direction.
  double h[DIM];
  // The width of the absorbing layer in each direction.
  double absorber_lens_neg[DIM], absorber_lens_pos[DIM];

  // dz = (1 + i sigma(x) / omega) dx.
  // sigma(x) = c (ratio)^2 / absorber_len.
  double pml_c;

  // Coordinate DMDA.
  DM cdm;
  Vec vcoords;
  // The main DM object.
  DM dm;

  // DM information.
  PetscInt x_start, y_start, z_start;
  PetscInt x_len, y_len, z_len;

  static std::complex<double> get_g(const double r, const double omega,
                                    const double c,
                                    const double absorber_len_neg,
                                    const double interior_domain_len,
                                    const double absorber_len_pos);

  PetscErrorCode _setup();

public:
  PetscErrorCode get_dm(DM *dm_out);

  PetscErrorCode get_vec_from_func(Vec v, func_ptr f, void *ctx);

  // Load Vec (size of interior_elems) from HDF5 file.
  // The dataset is in the location hdf5_groupname/vec_int_name.
  PetscErrorCode read_hdf5_vec(Vec v, const char *hdf5_filename,
                               const char *hdf5_groupname,
                               const char *vec_int_name);

  PetscErrorCode get_laplace_pml_mat(Mat A, const double omega);

  PetscErrorCode get_laplace_abc_mat(Mat A, const double omega);

  // PetscErrorCode get_laplace_cap_mat(Mat A, const double omega);

  PetscErrorCode get_delta_rhs(Vec rhs);

  // Will not zero the rhs first.
  PetscErrorCode get_delta_rhs(Vec rhs, PetscInt i0_offset, PetscInt j0_offset,
                               PetscInt k0_offset);

  PetscErrorCode print_info(const double omega);

  PetscErrorCode get_zeroed_boundary_vec(Vec v);

  // The .hdf5 and .xmf files will be save in the DATA_FOLDERPATH.
  // The dataset of the Petsc vector will be saved in the location
  // hdf5_groupname/Vec_name. The .xmf filename will be Vec_name +
  // xdmf_filename_surffix.
  PetscErrorCode save_xdmf_hdf5(Vec v, const char *xdmf_filename_surffix,
                                const char *hdf5_filename,
                                const char *hdf5_groupname);

  Solver(const int uniform_interior_elems, const int uniform_absorber_elems);

  Solver(const int uniform_absorber_elems, const int interior_elems[],
         const double interior_domain_lens[]);

  // The total_dofs = 2^levels + 1.
  // The interior_elems ~ total_dofs * ratio (should be an even number).
  // For testing multigrid.
  Solver(const int levels, const double ratio);

  ~Solver();
};

// A -> A + alpha / v^2 Id.
PetscErrorCode get_shifted_velocity_mat(Mat A, Vec v, const PetscScalar alpha);

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

// Complex shift preconditioner.
// Shift is a complex number.
struct ComplexShiftPre {
  // P = - (shift) omega^2/v^2 - Delta.
  double shift;
  double omega;
  Vec velocity;
  Mat P_mat;
  KSP P_ksp;
};
extern PetscErrorCode PCSetUp_ComplexShiftPre(PC pc);
extern PetscErrorCode PCApply_ComplexShiftPre(PC pc, Vec in, Vec out);
extern PetscErrorCode PCDestroy_ComplexShiftPre(PC pc);
PetscErrorCode PCShell_ComplexShiftPre(PC pc, ComplexShiftPre *ctx);

// Matrix exponential preconditioner.
// -i/T \int_0^T \int_0^t e^{i A s} ds dt ~ A^{-1}.
struct MatExPre {
  double delta_t;
  PetscInt steps;
  // Three matrix functions.
  MFN phi0; // aka exp
  MFN phi1;
  MFN phi2;
};
extern PetscErrorCode PCSetUp_MatExPre(PC pc);
extern PetscErrorCode PCApply_MatExPre(PC pc, Vec in, Vec out);
extern PetscErrorCode PCDestroy_MatExPre(PC pc);
PetscErrorCode PCShell_MatExPre(PC pc, MatExPre *ctx);

// Borrowed from https://petsc.org/main/src/ksp/ksp/tutorials/ex42.c.html.
PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_finest);

struct MatExPreVer2Ctx {
  double delta_t;
  PetscInt steps;
  double shift;
  double omega;
  Vec velocity;
  Mat P_mat;
  MFN phi0;
  MFN phi1;
};

extern PetscErrorCode PCSetUp_MatExPreVer2(PC pc);
extern PetscErrorCode PCApply_MatExPreVer2(PC pc, Vec in, Vec out);
extern PetscErrorCode PCDestroy_MatExPreVer2(PC pc);
extern PetscErrorCode PCShell_MatExPreVer2(PC pc, MatExPreVer2Ctx *ctx);