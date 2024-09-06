#include "gsl/gsl_errno.h"
#include "gsl/gsl_sf_elljac.h"
#include <cmath>
#include <iostream>
#include <petscsys.h>

int main(int argc, char **argv) {
  std::cout << "Hello World!\n";
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "This is a PETSc hello world!"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Hello world!\n"));

  PetscReal cn = 0.0, sn = 0.0, dn = 0.0, u = 0.3,
            one_over_sqrt2 = 1.0 / std::sqrt(2.0);
  auto gsl_status = gsl_sf_elljac_e(u, one_over_sqrt2, &sn, &cn, &dn);
  PetscAssert(gsl_status != GSL_EDOM, PETSC_COMM_WORLD,
              "Fail to performing cn(%.5e, %.5e)", u, one_over_sqrt2);

  PetscPrintf(PETSC_COMM_WORLD, "cn(%.5e, %.5e) = %.5e\n", u, one_over_sqrt2,
              cn);

  PetscInt m = -1, M = 7;
  PetscPrintf(PETSC_COMM_WORLD, "%d mod %d = %d\n", m, M, m % M);
  m = 7;
  PetscPrintf(PETSC_COMM_WORLD, "%d mod %d = %d\n", m, M, m % M);

  PetscFinalize();
  return 0;
}