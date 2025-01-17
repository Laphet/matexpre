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
  if (0.0 <= r && r <= interior_domain_len)
    return std::complex<double>(1.0, 0.0);
  else if (r < 0.0) {
    double temp = -r / absorber_len_neg;
    return 1.0 + IU * (c * temp * temp / absorber_len_neg) / omega;
  } else {
    double temp = (r - interior_domain_len) / absorber_len_pos;
    return 1.0 + IU * (c * temp * temp / absorber_len_pos) / omega;
  }
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
                           DMDA_STENCIL_STAR, total_dofs[0], total_dofs[1],
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
                           DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, total_dofs[0],
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
            (get_g(x, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_lens_pos[0]) *
             get_g(x_mhalf, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_elems_pos[0]));
        std::complex<double> temp_b =
            1.0 / (hx * hx) /
            (get_g(x, omega, pml_c, absorber_lens_neg[0],
                   interior_domain_lens[0], absorber_lens_pos[0]) *
             get_g(x_phalf, omega, pml_c, absorber_lens_pos[0],
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

  for (unsigned int i = 0; i < DIM; ++i) {
    t_ADITIONAL_INFO +=
        std::to_string(absorber_elems_neg[i] - 1) + " " +
        std::to_string(absorber_elems_neg[i] + interior_elems[i]);
    t_GRID_DIMENSIONS += std::to_string(total_dofs[i]);
    t_GRID_ORIGIN += std::to_string(static_cast<float>(-absorber_lens_neg[i]));
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
  PetscCall(KSPGetPC(ctx->P_ksp, &P_pc));
  // Set default KSP type.
  PetscCall(KSPSetType(ctx->P_ksp, KSPBCGS));
  // Allow CML options.
  PetscCall(PCSetOptionsPrefix(P_pc, "csp_"));
  PetscCall(KSPSetOptionsPrefix(ctx->P_ksp, "csp_"));
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
extern PetscErrorCode PCSetUp_MatExPre(PC pc) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  Mat A = nullptr;
  PC Z_pc = nullptr;
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  // Handle the CML options.
  PetscCall(PetscOptionsGetScalar(nullptr, nullptr, "-matex_alpha", &ctx->alpha,
                                  nullptr));

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_steps", &ctx->steps,
                               nullptr));
  PetscCheck(ctx->steps >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The steps must be at least 1, but got %d.\n", ctx->steps);
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-matex_time_steps_per_period",
                               &ctx->time_steps_per_period, nullptr));
  PetscCheck(ctx->time_steps_per_period >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE,
             "The time steps per period must be at least 1, but got %d.\n",
             ctx->time_steps_per_period);

  // Construct (-2i omega / delta_t alpha - omega^2 (1-alpha))/v^2 - Delta.
  PetscCall(PCGetOperators(pc, &A, nullptr));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx->Z_mat));
  // -omega^2 (1-alpha) + omega^2.
  PetscScalar shift = ctx->omega * ctx->omega * ctx->alpha;
  // -i omega / delta_t alpha.
  double delta_t = 2.0 * PETSC_PI / (ctx->omega * ctx->time_steps_per_period);
  shift -= 2.0 * IU * ctx->omega / delta_t * ctx->alpha;
  // Set the shifted matrix.
  PetscCall(get_shifted_velocity_mat(ctx->Z_mat, ctx->velocity, shift));

  // Set up the KSP for Z^{-1} b = u.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ctx->Z_ksp));
  PetscCall(KSPSetOperators(ctx->Z_ksp, ctx->Z_mat, ctx->Z_mat));
  PetscCall(KSPGetPC(ctx->Z_ksp, &Z_pc));
  // Set default KSP type.
  PetscCall(KSPSetType(ctx->Z_ksp, KSPBCGS));
  KSPSetInitialGuessNonzero(ctx->Z_ksp, PETSC_TRUE);
  // Allow CML options.
  PetscCall(PCSetOptionsPrefix(Z_pc, "matex_"));
  PetscCall(KSPSetOptionsPrefix(ctx->Z_ksp, "matex_"));
  PetscCall(KSPSetFromOptions(ctx->Z_ksp));
  // Set the PCMG for the Z_ksp.
  PetscBool use_pcmg = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(Z_pc), PCMG,
                                   &use_pcmg));
  DM dm = nullptr;
  PetscCall(VecGetDM(ctx->velocity, &dm));
  if (use_pcmg) {
    PetscCall(PCMGSetupViaCoarsen(Z_pc, dm));
  }
  // KSPSetUp setup will call PCSetUp.
  PetscCall(KSPSetUp(ctx->Z_ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_MatExPre(PC pc, Vec in, Vec out) {
  // Stack variables.
  MatExPre *ctx = nullptr;
  // Use CU here to keep sol as the initial guess for the next iteration.
  Vec rhs = nullptr, CU = nullptr, sol = nullptr;
  DM dm = nullptr;
  auto expim = [&](const double x) { return std::cos(x) + IU * std::sin(x); };
  PetscFunctionBeginUser;

  PetscCall(PCShellGetContext(pc, reinterpret_cast<void **>(&ctx)));
  double delta_t = 2.0 * PETSC_PI / (ctx->omega * ctx->time_steps_per_period);
  // Get the DM through the velocity.
  PetscCall(VecGetDM(ctx->velocity, &dm));

  PetscCall(VecZeroEntries(out));
  // Create temporary vectors.
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(DMGetGlobalVector(dm, &CU));
  PetscCall(DMGetGlobalVector(dm, &sol));
  PetscCall(VecZeroEntries(sol));
  for (unsigned int i = 1; i <= ctx->steps; ++i) {
    // CU -> out(U) / v^2.
    PetscCall(VecPointwiseDivide(CU, out, ctx->velocity));
    PetscCall(VecPointwiseDivide(CU, CU, ctx->velocity));
    // rhs -> in(f)
    PetscCall(VecCopy(in, rhs));
    // rhs -> -2i omega / delta_t * alpha * CU + exp(-i omega t^(k+0.5)) rhs.
    double t = (i + 0.5) * delta_t;
    PetscCall(VecAXPBY(rhs, -2.0 * IU * ctx->omega / delta_t * ctx->alpha,
                       expim(-ctx->omega * i * delta_t), CU));
    // Solve Z sol = rhs, sol ~ (U^{k+1}+U^k) / 2
    PetscCall(KSPSolve(ctx->Z_ksp, rhs, sol));
    // out(U) = -out + 2 sol
    PetscCall(VecAXPBY(out, 2.0, -1.0, sol));
  }
  // out = out exp(i omega T)
  double T = ctx->steps * delta_t;
  PetscCall(VecScale(out, expim(ctx->omega * T)));

  // Destroy the temporary vectors.
  PetscCall(DMRestoreGlobalVector(dm, &sol));
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

PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_finest) {
  // Stack variables.
  std::vector<DM> da_hierarchy;

  PetscFunctionBeginUser;
  PetscInt nlevels = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-pc_mg_levels", &nlevels, nullptr));
  PetscCheck(nlevels >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "The number of levels must be at least 2, but got %d.\n", nlevels);

  // Finest is at 0.
  da_hierarchy.resize(nlevels);
  da_hierarchy[0] = da_finest;
  PetscCall(DMCoarsenHierarchy(da_hierarchy[0], nlevels - 1, &da_hierarchy[1]));

  PetscCall(PCMGSetLevels(pc, nlevels, nullptr));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_PMAT));

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
  for (auto k = 0; k < nlevels - 1; ++k)
    PetscCall(DMDestroy(&da_hierarchy[k]));

  // Tests.
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD,
  //                       "PCMGSetupViaCoarsen is called with levels=%d.\n",
  //                       nlevels));

  PetscFunctionReturn(PETSC_SUCCESS);
}
