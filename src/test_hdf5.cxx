#include "petscdm.h"
#include "petscdmda.h"
#include "petscerror.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "petscviewerhdf5.h"

const int NX = 801;
const int NY = 801;
const int NZ = 187;

int main(int argc, char **argv) {

  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

  DM dm = nullptr;
  Vec v = nullptr;

  PetscInt pml_width = 10;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                         NX + 2 * pml_width, NY + 2 * pml_width,
                         NZ + 2 * pml_width, PETSC_DECIDE, PETSC_DECIDE,
                         PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecSet(v, 1.0));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(v), "velocity"));

  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "test.hdf5", FILE_MODE_APPEND,
                                &viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
}