# 2025-01-10
main in test_solver.cxx
> mpiexec -n 16 ./main -k 40 -absorber_elems 10
  Linear solve converged due to CONVERGED_RTOL iterations 926
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=926, relative residual norm=1.96452e-05, source norm=7.08706e+01, residual norm=1.39227e-03.

> mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_csp -csp_ksp_type preonly -csp_pc_type lu -ksp_monitor_true_residual
  0 KSP preconditioned resid norm 1.118150972746e-03 true resid norm 7.087061061454e+01 ||r(i)||/||b|| 1.000000000000e+00
  1 KSP preconditioned resid norm 9.448939050761e-07 true resid norm 1.942172915516e-02 ||r(i)||/||b|| 2.740448965621e-04
  2 KSP preconditioned resid norm 7.892797091279e-07 true resid norm 5.770737749666e-03 ||r(i)||/||b|| 8.142638675787e-05
  3 KSP preconditioned resid norm 6.725269831612e-07 true resid norm 1.216154419150e-02 ||r(i)||/||b|| 1.716020799883e-04
  4 KSP preconditioned resid norm 5.883183114736e-07 true resid norm 4.336204444006e-03 ||r(i)||/||b|| 6.118480434140e-05
  5 KSP preconditioned resid norm 5.423053502632e-07 true resid norm 6.621848518650e-03 ||r(i)||/||b|| 9.343574806580e-05
  6 KSP preconditioned resid norm 4.895955390413e-07 true resid norm 3.701315002551e-03 ||r(i)||/||b|| 5.222637381639e-05
  7 KSP preconditioned resid norm 4.523376185021e-07 true resid norm 4.453144905176e-03 ||r(i)||/||b|| 6.283486012836e-05
  8 KSP preconditioned resid norm 3.789746522362e-07 true resid norm 3.719492704558e-03 ||r(i)||/||b|| 5.248286521458e-05
  9 KSP preconditioned resid norm 2.668327084705e-07 true resid norm 3.667067638156e-03 ||r(i)||/||b|| 5.174313592557e-05
 10 KSP preconditioned resid norm 1.278330229140e-07 true resid norm 2.310320149269e-03 ||r(i)||/||b|| 3.259912859838e-05
 11 KSP preconditioned resid norm 4.753086402421e-08 true resid norm 1.022145485186e-03 ||r(i)||/||b|| 1.442269900489e-05
 12 KSP preconditioned resid norm 1.500706456899e-08 true resid norm 3.783534503644e-04 ||r(i)||/||b|| 5.338650917264e-06
 13 KSP preconditioned resid norm 4.277786067474e-09 true resid norm 1.236445379953e-04 ||r(i)||/||b|| 1.744651794632e-06
  Linear solve converged due to CONVERGED_RTOL iterations 13
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=13, relative residual norm=1.74465e-06, source norm=7.08706e+01, residual norm=1.23645e-04.

> mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_matex -matex_ksp_type preonly -matex_pc_type lu -ksp_monitor_true_residual