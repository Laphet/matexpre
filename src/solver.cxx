#include "solver.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscoptions.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewerhdf5.h"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <string>

template <unsigned int DIM>
std::complex<double> Solver<DIM>::get_g(const double r, const double omega,
                                        const double c,
                                        const double absorber_len,
                                        const double interior_domain_len) {
  if (0.0 <= r && r <= interior_domain_len)
    return std::complex<double>(1.0, 0.0);
  else {
    double temp =
        r < 0.0 ? -r / absorber_len : (r - interior_domain_len) / absorber_len;
    return 1.0 + IU * (c * temp * temp / absorber_len) / omega;
  }
}

template <unsigned int DIM> PetscErrorCode Solver<DIM>::_setup() {
  // Receive the CML arguments.
  PetscReal pml_c_uniform = 0.0;
  PetscBool received_pml_c_uniform = PETSC_FALSE;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-pml_c_uniform",
                                &pml_c_uniform, &received_pml_c_uniform));
  if (received_pml_c_uniform) {
    PetscCheck(pml_c_uniform >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "pml_c must be non-negative, but got %f.", pml_c_uniform);
    for (unsigned int i = 0; i < DIM; ++i)
      pml_c[i] = pml_c_uniform;
  }

  for (unsigned int i = 0; i < DIM; ++i) {
    total_dofs[i] = interior_elems[i] + 2 * absorber_elems[i] - 1;
    h[i] = interior_domain_lens[i] / interior_elems[i];
    absorber_lens[i] = h[i] * absorber_elems[i];
  }

  PetscFunctionBeginUser;

  if constexpr (DIM == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR, total_dofs[0], total_dofs[1],
                           PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr,
                           &dm));

    PetscCall(DMSetUp(dm));

    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens[0] + h[0],
        interior_domain_lens[0] + absorber_lens[0] - h[0],
        -absorber_lens[1] + h[1],
        interior_domain_lens[1] + absorber_lens[1] - h[1], 0.0, 0.0));
  }

  if constexpr (DIM == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, total_dofs[0],
                           total_dofs[1], total_dofs[2], PETSC_DECIDE,
                           PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr,
                           nullptr, &dm));
    PetscCall(DMSetUp(dm));

    // Set the coordinates of the DMDA.
    PetscCall(DMDASetUniformCoordinates(
        dm, -absorber_lens[0] + h[0],
        interior_domain_lens[0] + absorber_lens[0] - h[0],
        -absorber_lens[1] + h[1],
        interior_domain_lens[1] + absorber_lens[1] - h[1],
        -absorber_lens[2] + h[2],
        interior_domain_lens[2] + absorber_lens[2] - h[2]));
  }

  // Get the coordinates.
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &vcoords));
  // Get DMDA information.
  PetscCall(
      DMDAGetCorners(dm, &x_start, &y_start, &z_start, &x_len, &y_len, &z_len));

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
        PetscReal x_coord = acoords_2d[y_ind][x_ind].x.real(),
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
          PetscReal x_coord = acoords_3d[z_ind][y_ind][x_ind].x.real(),
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
PetscErrorCode Solver<DIM>::get_laplace_mat(Mat mat, const double omega) {
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
            get_g(x, omega, pml_c[0], absorber_lens[0],
                  interior_domain_lens[0]) /
            get_g(x_mhalf, omega, pml_c[0], absorber_lens[0],
                  interior_domain_lens[0]);
        std::complex<double> temp_b =
            1.0 / (hx * hx) /
            get_g(x, omega, pml_c[0], absorber_lens[0],
                  interior_domain_lens[0]) /
            get_g(x_phalf, omega, pml_c[0], absorber_lens[0],
                  interior_domain_lens[0]);
        std::complex<double> temp_c =
            1.0 / (hy * hy) /
            get_g(y, omega, pml_c[1], absorber_lens[1],
                  interior_domain_lens[1]) /
            get_g(y_mhalf, omega, pml_c[1], absorber_lens[1],
                  interior_domain_lens[1]);
        std::complex<double> temp_d =
            1.0 / (hy * hy) /
            get_g(y, omega, pml_c[1], absorber_lens[1],
                  interior_domain_lens[1]) /
            get_g(y_phalf, omega, pml_c[1], absorber_lens[1],
                  interior_domain_lens[1]);

        MatStencil row = {0, y_ind, x_ind, 0};
        MatStencil cols[5] = {{0, y_ind, x_ind, 0},
                              {0, y_ind, x_ind - 1, 0},
                              {0, y_ind, x_ind + 1, 0},
                              {0, y_ind - 1, x_ind, 0},
                              {0, y_ind + 1, x_ind, 0}};
        PetscScalar vals[5] = {temp_a + temp_b + temp_c + temp_d, -temp_a,
                               -temp_b, -temp_c, -temp_d};

        PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[0], &vals[0],
                                      INSERT_VALUES));
        if (x_ind - 1 >= 0)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[1], &vals[1],
                                        INSERT_VALUES));
        if (x_ind + 1 < total_dofs[0])
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[2], &vals[2],
                                        INSERT_VALUES));
        if (y_ind - 1 >= 0)
          PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[3], &vals[3],
                                        INSERT_VALUES));
        if (y_ind + 1 < total_dofs[1])
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
              get_g(x, omega, pml_c[0], absorber_lens[0],
                    interior_domain_lens[0]) /
              get_g(x_mhalf, omega, pml_c[0], absorber_lens[0],
                    interior_domain_lens[0]);
          std::complex<double> temp_b =
              1.0 / (hx * hx) /
              get_g(x, omega, pml_c[0], absorber_lens[0],
                    interior_domain_lens[0]) /
              get_g(x_phalf, omega, pml_c[0], absorber_lens[0],
                    interior_domain_lens[0]);
          std::complex<double> temp_c =
              1.0 / (hy * hy) /
              get_g(y, omega, pml_c[1], absorber_lens[1],
                    interior_domain_lens[1]) /
              get_g(y_mhalf, omega, pml_c[1], absorber_lens[1],
                    interior_domain_lens[1]);
          std::complex<double> temp_d =
              1.0 / (hy * hy) /
              get_g(y, omega, pml_c[1], absorber_lens[1],
                    interior_domain_lens[1]) /
              get_g(y_phalf, omega, pml_c[1], absorber_lens[1],
                    interior_domain_lens[1]);
          std::complex<double> temp_e =
              1.0 / (hz * hz) /
              get_g(z, omega, pml_c[2], absorber_lens[2],
                    interior_domain_lens[2]) /
              get_g(z_mhalf, omega, pml_c[2], absorber_lens[2],
                    interior_domain_lens[2]);
          std::complex<double> temp_f =
              1.0 / (hz * hz) /
              get_g(z, omega, pml_c[2], absorber_lens[2],
                    interior_domain_lens[2]) /
              get_g(z_phalf, omega, pml_c[2], absorber_lens[2],
                    interior_domain_lens[2]);

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
          if (x_ind - 1 >= 0)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[1], &vals[1],
                                          INSERT_VALUES));
          if (x_ind + 1 < total_dofs[0])
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[2], &vals[2],
                                          INSERT_VALUES));
          if (y_ind - 1 >= 0)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[3], &vals[3],
                                          INSERT_VALUES));
          if (y_ind + 1 < total_dofs[1])
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[4], &vals[4],
                                          INSERT_VALUES));
          if (z_ind - 1 >= 0)
            PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &cols[5], &vals[5],
                                          INSERT_VALUES));
          if (z_ind + 1 < total_dofs[2])
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
    absorber_elems[i] = uniform_absorber_elems;
    pml_c[i] = 20.0;
  }
  PetscCallAbort(PETSC_COMM_SELF, _setup());
}

template <unsigned int DIM> PetscErrorCode Solver<DIM>::print_info() {
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Absorber elements: Nx=%d, Ny=%d",
                        absorber_elems[0], absorber_elems[1]));
  if constexpr (DIM == 3) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", Nz=%d.\n", absorber_elems[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PML constants: c_x=%.5f, c_y=%.5f",
                        pml_c[0], pml_c[1]));
  if constexpr (DIM == 3) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", c_z=%.5f.\n", pml_c[2]));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, ".\n"));
  }

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

  for (unsigned int i = 0; i < DIM; ++i) {
    t_ADITIONAL_INFO += std::to_string(absorber_elems[i] - 1) + " " +
                        std::to_string(absorber_elems[i] + interior_elems[i]);
    t_GRID_DIMENSIONS += std::to_string(total_dofs[i]);
    t_GRID_ORIGIN += std::to_string(static_cast<float>(-absorber_lens[i]));
    t_GRID_SPACING += std::to_string(static_cast<float>(h[i]));
    if (i != DIM - 1) {
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
  xdmf_full_filename += std::string(vec_name) +
                        std::string(xdmf_filename_surffix) +
                        std::string(".xmf");
  // Save .hdf5 file.
  PetscViewer hdf5_viewer = nullptr;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, hdf5_full_filename.c_str(),
                                FILE_MODE_WRITE, &hdf5_viewer));
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
  // Construct P = A - shift Iu.
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-csp_shift", &ctx->shift,
                                nullptr));
  PetscCheck(ctx->shift >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The shift must be non-negative, but got %f.\n", ctx->shift);

  PetscCall(PCGetOperators(pc, &A, nullptr));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx->P_mat));
  PetscCall(get_shifted_velocity_mat(
      ctx->P_mat, ctx->velocity, -ctx->omega * ctx->omega * IU * ctx->shift));
  // Set up the KSP for P^{-1} b = u.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ctx->P_ksp));
  PetscCall(KSPSetOperators(ctx->P_ksp, ctx->P_mat, ctx->P_mat));
  // Set the initial guess to be nonzero, through the input vector.
  // PetscCall(KSPSetInitialGuessNonzero(ctx->P_ksp, PETSC_TRUE));
  PetscCall(KSPGetPC(ctx->P_ksp, &P_pc));
  // Allow CML options.
  PetscCall(PCSetOptionsPrefix(P_pc, "csp_"));
  PetscCall(KSPSetOptionsPrefix(ctx->P_ksp, "csp_"));
  PetscCall(KSPSetFromOptions(ctx->P_ksp));
  // KSPSetUp setup will call PCSetUp.
  PetscCall(KSPSetUp(ctx->P_ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_ComplexShiftPre(PC pc, Vec in, Vec out) {
  // Stack variables.
  ComplexShiftPre *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(KSPSolve(ctx->P_ksp, in, out));

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
extern PetscErrorCode PCSetUp_MatExPre(PC pc) { // Stack variables.
  MatExPre *ctx = nullptr;
  Mat A = nullptr;
  PC Z_pc = nullptr;
  PetscScalar shift = 0.0;
  PetscReal delta_t = 0.0;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  // Handle the CML options.
  PetscCall(PetscOptionsGetScalar(nullptr, nullptr, "-matex_alpha", &ctx->alpha,
                                  nullptr));

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_periods",
                               &ctx->periods, nullptr));
  PetscCheck(ctx->periods >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The periods must be at least 1, but got %d.\n", ctx->periods);
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_time_steps_per_period",
                               &ctx->time_steps_per_period, nullptr));
  PetscCheck(ctx->time_steps_per_period >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE,
             "The time steps per period must be at least 1, but got %d.\n",
             ctx->time_steps_per_period);

  // Construct (-i omega / delta_t alpha - omega^2 (1-alpha))/v^2 - Delta.
  PetscCall(PCGetOperators(pc, &A, nullptr));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx->Z_mat));
  // -omega^2 (1-alpha) + omega^2.
  shift = ctx->omega * ctx->omega * ctx->alpha;
  // -i omega / delta_t alpha.
  delta_t = 2.0 * PETSC_PI / (ctx->omega * ctx->time_steps_per_period);
  shift -= IU * ctx->omega / delta_t * ctx->alpha;
  // Set the shifted matrix.
  PetscCall(get_shifted_velocity_mat(ctx->Z_mat, ctx->velocity, shift));

  // Set up the KSP for Z^{-1} b = u.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ctx->Z_ksp));
  PetscCall(KSPSetOperators(ctx->Z_ksp, ctx->Z_mat, ctx->Z_mat));
  PetscCall(KSPGetPC(ctx->Z_ksp, &Z_pc));
  // Allow CML options.
  PetscCall(PCSetOptionsPrefix(Z_pc, "matex_"));
  PetscCall(KSPSetOptionsPrefix(ctx->Z_ksp, "matex_"));
  PetscCall(KSPSetFromOptions(ctx->Z_ksp));
  // KSPSetUp setup will call PCSetUp.
  PetscCall(KSPSetUp(ctx->Z_ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_MatExPre(PC pc, Vec in, Vec out) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  // Use CU here to keep sol as the initial guess for the next iteration.
  Vec rhs = nullptr, CU = nullptr;
  DM dm = nullptr;
  PetscReal delta_t = 0.0;
  auto expim = [&](const double x) { return std::cos(x) + IU * std::sin(x); };
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  delta_t = 2.0 * PETSC_PI / (ctx->omega * ctx->time_steps_per_period);
  // Get the DM through the velocity.
  PetscCall(VecGetDM(ctx->velocity, &dm));

  PetscCall(VecZeroEntries(out));
  // Create temporary vectors.
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(DMGetGlobalVector(dm, &CU));
  for (unsigned int i = 1; i <= ctx->time_steps_per_period * ctx->periods;
       ++i) {
    PetscReal t = i * delta_t;
    // rhs -> 2exp(-i omega t^(k+0.5)) f.
    PetscCall(VecCopy(in, rhs));
    PetscCall(VecScale(rhs, expim(-ctx->omega * i * delta_t)));
    // CU -> U / v^2.
    PetscCall(VecPointwiseDivide(CU, out, ctx->velocity));
    PetscCall(VecPointwiseDivide(CU, CU, ctx->velocity));
    // rhs -> rhs + -i omega / detla_t * alpha CU.
    PetscCall(VecAXPY(rhs, -4.0 * IU * ctx->omega * ctx->alpha / delta_t, CU));
    // Solve Z U = rhs.
    PetscCall(KSPSolve(ctx->Z_ksp, rhs, out));
  }

  // Destroy the temporary vectors.
  PetscCall(DMRestoreGlobalVector(dm, &CU));
  PetscCall(DMRestoreGlobalVector(dm, &rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_MatExPre(PC pc) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  PetscCall(MatDestroy(&ctx->Z_mat));
  // KSPDestroy will call PCDestroy.
  PetscCall(KSPDestroy(&ctx->Z_ksp));

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
