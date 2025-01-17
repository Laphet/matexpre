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

# 2025-01-13
I changed the Schrodinger equation to
 -i omega dot(U) alpha - omega^2 U (1-alpha) - v^2 Delta U = g exp(-i omega t)
Set alpha = 1 / omega

mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_matex -matex_ksp_type preonly -matex_pc_type lu -ksp_monitor_true_residual
  0 KSP preconditioned resid norm 1.116873375129e-03 true resid norm 7.087061061454e+01 ||r(i)||/||b|| 1.000000000000e+00
  1 KSP preconditioned resid norm 2.824523098632e-06 true resid norm 3.009122538315e-03 ||r(i)||/||b|| 4.245938495833e-05
  2 KSP preconditioned resid norm 2.667022405323e-07 true resid norm 8.087113658644e-04 ||r(i)||/||b|| 1.141109634659e-05
  3 KSP preconditioned resid norm 1.847049654003e-08 true resid norm 1.072438066459e-04 ||r(i)||/||b|| 1.513233845679e-06
  4 KSP preconditioned resid norm 1.016827239460e-09 true resid norm 9.532868644345e-06 ||r(i)||/||b|| 1.345108862712e-07
  Linear solve converged due to CONVERGED_RTOL iterations 4
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=4, relative residual norm=1.34511e-07, source norm=7.08706e+01, residual norm=9.53287e-06.

# 2025-01-14
## Test the Laplace problem eigenvalues.
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -pc_type gamg
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=145, relative residual norm=2.95952e-03, source norm=7.08706e+01, residual norm=2.09743e-01.
Number of converged eigenpairs: 8
Eigenvalue 0: 1.27998e+06       +       -1.52261e-01i
Eigenvalue 1: 1.27995e+06       +       -3.80662e-01i
Eigenvalue 2: 1.27992e+06       +       -6.09049e-01i
Eigenvalue 3: 1.27990e+06       +       -7.61326e-01i
Eigenvalue 4: 1.27987e+06       +       -9.89744e-01i
Eigenvalue 5: 1.27984e+06       +       -1.29432e+00i
Eigenvalue 6: 1.27983e+06       +       -1.37042e+00i
Eigenvalue 7: 1.27981e+06       +       -1.52274e+00i

mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_largest_real
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=3270, relative residual norm=1.04128e-05, source norm=7.08706e+01, residual norm=7.37959e-04.
Number of converged eigenpairs: 8
Eigenvalue 0: 1.27998e+06       +       -1.52261e-01i
Eigenvalue 1: 1.27995e+06       +       -3.80652e-01i
Eigenvalue 2: 1.27992e+06       +       -6.09051e-01i
Eigenvalue 3: 1.27990e+06       +       -7.61321e-01i
Eigenvalue 4: 1.27987e+06       +       -9.89743e-01i
Eigenvalue 5: 1.27984e+06       +       -1.29431e+00i
Eigenvalue 6: 1.27983e+06       +       -1.37042e+00i
Eigenvalue 7: 1.27981e+06       +       -1.52273e+00i

mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_smallest_real
  Linear solve converged due to CONVERGED_RTOL iterations 3270
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=3270, relative residual norm=1.04128e-05, source norm=7.08706e+01, residual norm=7.37959e-04.
Number of converged eigenpairs: 8
Eigenvalue 0: -3.22086e+04      +       -1.39906e+05i
Eigenvalue 1: -3.22086e+04      +       -1.39906e+05i
Eigenvalue 2: -3.22086e+04      +       -1.39906e+05i
Eigenvalue 3: -3.22086e+04      +       -1.39906e+05i
Eigenvalue 4: -1.60954e+04      +       -6.99540e+04i
Eigenvalue 5: -1.60688e+04      +       -6.99567e+04i
Eigenvalue 6: -1.60244e+04      +       -6.99612e+04i
Eigenvalue 7: -1.59622e+04      +       -6.99675e+04i

mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_smallest_real -pml_c_uniform 10.0
  Linear solve converged due to CONVERGED_RTOL iterations 991
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=10.00000, c_y=10.00000.
Number of iterations=991, relative residual norm=1.04445e-05, source norm=7.08706e+01, residual norm=7.40207e-04.
Number of converged eigenpairs: 9
Eigenvalue 0: 1.78699e+01       +       -9.01196e-01i
Eigenvalue 1: 4.46743e+01       +       -2.25296e+00i
Eigenvalue 2: 4.46743e+01       +       -2.25296e+00i
Eigenvalue 3: 7.14787e+01       +       -3.60472e+00i
Eigenvalue 4: 8.93467e+01       +       -4.50578e+00i
Eigenvalue 5: 8.93467e+01       +       -4.50578e+00i
Eigenvalue 6: 1.16151e+02       +       -5.85754e+00i
Eigenvalue 7: 1.16151e+02       +       -5.85754e+00i
Eigenvalue 8: 1.51885e+02       +       -7.65948e+00i

It seems that gamg does not actually converge.

# 2025-01-15
Those commands work.

./main -ksp_type fgmres -use_csp -csp_ksp_max_it 2 -csp_pc_type mg

./main -ksp_type fgmres -use_csp -csp_ksp_rtol 1.0e-1 -csp_pc_type mg

./main -ksp_type fgmres -use_csp -csp_pc_type mg  -csp_ksp_rtol 1.0e-1 -csp_mg_levels_0_ksp_type preonly -csp_mg_levels_0_pc_type lu

# 2015-01-16
Found using asm will improve the performance.
./main -ksp_max_it 50 -use_csp -csp_ksp_type preonly -csp_pc_type mg  -csp_ksp_monitor_true_residual -csp_shift 0.5 -csp_pc_mg_cycle_type v -pc_mg_levels 3 -csp_mg_levels_pc_type asm -csp_mg_levels_ksp_type bcgs
  iter 25
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm
  iter 85
  Average inner mg iterations 2
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.5 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 42
  Average inner mg iterations 2
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.2 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 39
  Average inner mg iterations 3
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.1 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 16
  Average inner mg iterations 8
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.05 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 7
  Average inner mg iterations 15
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 2
  Average inner mg iterations 15
mpiexec -n 16 ./main -k 60 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 2
  Average inner mg iterations 18
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
  iter 2
  Average inner mg iterations 35
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 0.1 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 2 -csp_ksp_converged_reason
  iter 4
  Average inner mg iterations 10
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 3 -csp_ksp_converged_reaso
  iter 6
  Average inner mg iterations 4
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 4 -csp_ksp_converged_reason
  iter 7
  Average inner mg iterations 4
mpiexec -n 16 ./main -k 100 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 4 -csp_ksp_converged_reason
  iter 8
  Average inner mg iterations 8
mpiexec -n 16 ./main -k 120 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 5 -csp_ksp_converged_reason
  iter 7
  Average inner mg iterations 6
mpiexec -n 16 ./main -k 140 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 5 -csp_ksp_converged_reason
  iter 8
  Average inner mg iterations 5














  
