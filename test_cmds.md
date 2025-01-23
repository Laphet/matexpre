# 2025-01-10
main in test_solver.cxx
``` 
mpiexec -n 16 ./main -k 40 -absorber_elems 10
```
  Linear solve converged due to CONVERGED_RTOL iterations 926
Interior domain lengths: Lx=1.00000, Ly=1.00000.
Interior elements: Nx=400, Ny=400.
Absorber elements: Nx=10, Ny=10.
PML constants: c_x=20.00000, c_y=20.00000.
Number of iterations=926, relative residual norm=1.96452e-05, source norm=7.08706e+01, residual norm=1.39227e-03.

``` 
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_csp -csp_ksp_type preonly -csp_pc_type lu -ksp_monitor_true_residual
```
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

```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_matex -matex_ksp_type preonly -matex_pc_type lu -ksp_monitor_true_residual
```

# 2025-01-13
I changed the Schrodinger equation to
 -i omega dot(U) alpha - omega^2 U (1-alpha) - v^2 Delta U = g exp(-i omega t)
Set alpha = 1 / omega

```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -use_matex -matex_ksp_type preonly -matex_pc_type lu -ksp_monitor_true_residual
```
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
```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -pc_type gamg
```
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

```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_largest_real
```
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

```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_smallest_real
```
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

```
mpiexec -n 16 ./main -k 40 -absorber_elems 10 -eps_smallest_real -pml_c_uniform 10.0
```
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

```
./main -ksp_type fgmres -use_csp -csp_ksp_max_it 2 -csp_pc_type mg
```

``` 
./main -ksp_type fgmres -use_csp -csp_ksp_rtol 1.0e-1 -csp_pc_type mg
```

```
./main -ksp_type fgmres -use_csp -csp_pc_type mg  -csp_ksp_rtol 1.0e-1 -csp_mg_levels_0_ksp_type preonly -csp_mg_levels_0_pc_type lu
```

# 2025-01-16
Found using asm will improve the performance.

```
./main -ksp_max_it 50 -use_csp -csp_ksp_type preonly -csp_pc_type mg  -csp_ksp_monitor_true_residual -csp_shift 0.5 -csp_pc_mg_cycle_type v -pc_mg_levels 3 -csp_mg_levels_pc_type asm -csp_mg_levels_ksp_type bcgs
```
  iter 25

```
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm
```
  iter 85
  Average inner mg iterations 2

```
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.5 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 42
  Average inner mg iterations 2

```
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.2 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 39
  Average inner mg iterations 3

```
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.1 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 16
  Average inner mg iterations 8

```
mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.05 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 7
  Average inner mg iterations 15

```mpiexec -n 16 ./main -k 40 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 2
  Average inner mg iterations 15

```
mpiexec -n 16 ./main -k 60 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 2
  Average inner mg iterations 18

```
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 0.0 -csp_ksp_type bcgs -csp_ksp_rtol 0.001 -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual
```
  iter 2
  Average inner mg iterations 35

```
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 0.1 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 2 -csp_ksp_converged_reason
```
  iter 4
  Average inner mg iterations 10

```
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 3 -csp_ksp_converged_reaso
```
  iter 6
  Average inner mg iterations 4

```
mpiexec -n 16 ./main -k 80 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 4 -csp_ksp_converged_reason
```
  iter 7
  Average inner mg iterations 4

```
mpiexec -n 16 ./main -k 100 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 4 -csp_ksp_converged_reason
```
  iter 8
  Average inner mg iterations 8

```
mpiexec -n 16 ./main -k 120 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 5 -csp_ksp_converged_reason
```
  iter 7
  Average inner mg iterations 6

``` 
mpiexec -n 16 ./main -k 140 -ksp_monitor_true_residual -use_csp -csp_shift 1.0 -csp_ksp_rtol 0.0001 -csp_ksp_max_it 10 -csp_ksp_type bcgs -csp_pc_type mg -csp_mg_levels_pc_type asm -csp_ksp_monitor_true_residual -pc_mg_levels 5 -csp_ksp_converged_reason
```
  iter 8
  Average inner mg iterations 5

# 2025-01-20
Found that if the rhs is specially chosen, the GMRES can converge in few iterations.
E.g., in our two-pole source term, if we set r=1/16, there only are 5 iterations needed.
Strange...

It seems something wrong with superlu_dist...

```
mpiexec -n 16 ./main -grids 9 -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist
```
It shows a very slow convergence.

```
mpiexec -n 16 ./main -grids 9 -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mkl_cpardiso
```
cpardiso is normal.

```
mpiexec -n 16 ./main -grids 8 -use_csp -csp_shift 0.5 -csp_ksp_rtol 0.01 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 3 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual
```
  iter 61
  Average inner mg iterations 5

```
mpiexec -n 16 ./main -grids 8 -use_csp -csp_shift 0.5 -csp_ksp_rtol 0.01 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 2 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual
```
  iter 57
  Average inner mg iterations 2

Found that csp_shift=0.5 for grids=9 is not enough. 

```
mpiexec -n 16 ./main -grids 9 -use_csp -csp_shift 0.5 -csp_ksp_rtol 0.0001 -csp_pc_type lu -csp_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual
```
  iter 118

```
mpiexec -n 16 ./main -grids 9 -use_csp -csp_shift 0.2 -csp_ksp_rtol 0.0001 -csp_pc_type lu -csp_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual
```
  iter 39


```
mpiexec -n 16 ./main -grids 8 -use_matex -matex_steps 4 -matex_time_steps_per_period 4 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_initial_guess_nonzero false
```

Let's go to matex.
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps (1-4) -matex_time_steps_per_period 4 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_initial_guess_nonzero false
```
  steps=1 iter=5
  steps=2 iter=6
  steps=3 iter=8
  steps=4 iter=1491

# 2025-01-21
```
mpiexec -n 16 ./main -grids 10 -use_matex -matex_steps (1-4) -matex_time_steps_per_period 4 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_initial_guess_nonzero false
```
  steps=1 iter=5
  steps=2 iter=5
  steps=3 iter=7
  steps=4 iter=?

```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period (1-8) -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_initial_guess_nonzero false
```
  steps_per_period=1 iter=4
  steps_per_period=2 iter=5
  steps_per_period=3 iter=5
  steps_per_period=4 iter=5
  steps_per_period=5 iter=5
  steps_per_period=6 iter=5
  steps_per_period=7 iter=6
  steps_per_period=8 iter=6
  steps_per_period=10 iter=6
  steps_per_period=16 iter=7
  steps_per_period=32 iter=12
  steps_per_period=64 iter=18

Compare csp with matex, what happened if the matex shift is applied on csp?
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_initial_guess_nonzero false
```
  iter=4

```
mpiexec -n 16 ./main -grids 9 -use_csp -csp_shift 96.0 -csp_pc_type lu -csp_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual
```
  iter>200? 
I think that csp wrongly guessed the phases of the Schrodinger equation.
Great news!

Test mg with matex.
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_pc_type mg -pc_mg_levels 3 -matex_mg_levels_pc_type asm -matex_mg_coarse_pc_type lu -matex_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_ksp_monitor_true_residual
```
Something wrong, it seems, alpha*steps should be constant to allow a reasonable mg convergence.
Therefore, omega*alpha/delta_t looks like omega^2.
What is the convergence of matex w.r.t. alpha?

Fair comparison
```
mpiexec -n 16 ./main -grids 9 -use_csp -csp_shift 0.15915 -csp_pc_type lu -csp_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual
```
iter 29
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.5
```
iter 449
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 2 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.25
```
iter 107
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 4 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.125
```
iter 50
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 8 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.0625
```
iter 32
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 2 -matex_time_steps_per_period 16 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.03125
```
iter >300
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 2 -matex_time_steps_per_period 8 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.0625
```
iter >200
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 14 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.02
```
iter 21
```
mpiexec -n 16 ./main -grids 9 -use_csp -csp_shift 0.08913 -csp_pc_type lu -csp_pc_fact
or_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual
```
iter 19
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 16 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.0125
```
iter 17


Fix matex_time_steps_period=1, vary alpha, which looks like csp.
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.2
```
  iter 147, csp_shift=0.06366, csp_iter 17
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 5 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.04
```
  iter 20, csp_shift=0.06366
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 10 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.02
```
  iter 18, csp_shift=0.06366
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 20 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.01
```
  iter 17, csp_shift=0.06366
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 40 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.005
```
  iter 17, csp_shift=0.06366

It seems that maxex_alpha is more important, first minimize alpha.
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 2 -matex_time_steps_per_period 40 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.005
```
  iter >400, csp_shift=0.06366
```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 2 -matex_time_steps_per_period 100 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.005
```
  iter 59, csp_shift=0.15915
```



```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.1
```
  iter 45, csp_shift=0.03183, csp_iter 10



```
mpiexec -n 16 ./main -grids 9 -use_matex -matex_steps 1 -matex_time_steps_per_period 1 -matex_ksp_type preonly -matex_pc_type lu -matex_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -matex_alpha 0.05
```
  iter 17, csp_shift=0.01592, csp_iter 7

Test mg, dose mg deterioate w.r.t. omega? first csp.
Find the sweet point of mg.
```
mpiexec -n 16 ./main -grids (6-11) -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 2 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=6, mg_iter=9
  grids=7, mg_iter=5
  grids=8, mg_iter=4
  grids=9, mg_iter=4
  grids=10, mg_iter=4
  grids=11, killed

```
mpiexec -n 16 ./main -grids (6-11) -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 3 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=6, mg_iter=55
  grids=7, mg_iter=99
  grids=8, mg_iter=16
  grids=9, mg_iter=15
  grids=10, mg_iter=17
  grids=11, mg_iter=22
```
mpiexec -n 16 ./main -grids (6-11) -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 4 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=6, mg_iter=45
  grids=7, mg_iter=141
  grids=8, mg_iter=173
  grids=9, mg_iter=547
  grids=10, mg_iter>1000
```
mpiexec -n 16 ./main -grids (6-11) -use_csp -csp_shift 1.0 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 4 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=6, mg_iter=27
  grids=7, mg_iter=61
  grids=8, mg_iter=64
  grids=9, mg_iter>1000
```
mpiexec -n 16 ./main -grids (6-11) -use_csp -csp_shift 2.0 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 4 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=6, mg_iter=4
  grids=7, mg_iter=4
  grids=8, mg_iter=4
  grids=9, mg_iter=4
  grids=10, mg_iter=4
  grids=11, mg_iter=4
```
mpiexec -n 16 ./main -grids (7-11) -use_csp -csp_shift 2.0 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 1
```
  grids=7, mg_iter=4
  grids=8, mg_iter=4
  grids=9, mg_iter=4
  grids=10, mg_iter=4
  grids=11, mg_iter=4
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.57079 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5 -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  grids=11, mg_iter=5
  without -csp_mg_levels_pc_type asm, mg does not converge. 

Found that "-csp_mg_levels_ksp_type richardson" converges much fast. This is normal, as the matrix is not SPD.
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.57079 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5 -csp_mg_levels_ksp_type richardson -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=2
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.57079 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5 -csp_mg_levels_ksp_type richardson -csp_mg_levels_pc_type bjacobi -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=3
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.57079 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5  -csp_mg_levels_ksp_type richardson -csp_mg_levels_ksp_max_it 1 -csp_mg_levels_pc_type bjacobi -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=3~5, seems to be faster.
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.57079 -csp_pc_type mg -csp_mg_levels_pc_type asm -pc_mg_levels 5  -csp_mg_levels_ksp_type bcgs -csp_mg_levels_ksp_max_it 1 -csp_mg_levels_pc_type bjacobi -csp_mg_coarse_pc_type lu -csp_mg_coarse_pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=2, why not bcgs?

# 2025-01-22
Now I changed the default mg setting, test which shift provide mg convergence.
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 1.0 -csp_pc_type mg -csp_pc_mg_levels 5 -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=2
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_pc_mg_levels 5 -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  mg_iter=3
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-2 -matex_time_steps_per_period 100 -matex_steps 1 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10
```
  Final error=2.024751250949e-01
```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_pc_mg_levels 5 -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10
```
  Final error=2.030101034210e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-2 -matex_time_steps_per_period 100 -matex_steps 2 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10
```
  Final error=1.54673e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-2 -matex_time_steps_per_period 100 -matex_steps 3 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero
```
  Final error=4.40646e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 1 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero
```
  Final error=2.02945e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 2 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero
```
  Final error=1.55560e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 3 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero
```
  Final error=4.23801e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 10 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero
```
  Final error=3.10278e-01

```
mpiexec -n 16 ./main -grids 11 -use_csp -csp_shift 0.5 -csp_pc_type mg -csp_pc_mg_levels 5 -ksp_monitor_true_residual -csp_ksp_monitor_true_residual -ksp_max_it 10 -csp_ksp_max_it 1
```
  Final error=2.04737e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 2 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=1.67710e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 4 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=3.23050e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-3 -matex_time_steps_per_period 1000 -matex_steps 8 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=2.43023e-01

It seems that use one mg for a ksp is enough.
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-1 -matex_time_steps_per_period 10 -matex_steps 1 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=2.11975e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-1 -matex_time_steps_per_period 10 -matex_steps 2 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=1.65188e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-1 -matex_time_steps_per_period 10 -matex_steps 3 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
  Final error=4.42683e-01
It seems pretty bad...

```
 mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-2 -matex_time_steps_per_period 100 -matex_steps 20 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=2.65646e-01
```
mpiexec -n 16 ./main -grids 11 -use_matex -matex_alpha 1.57079e-2 -matex_time_steps_per_period 100 -matex_steps 20 -matex_pc_type mg -matex_pc_mg_levels 5 -ksp_monitor_true_residual -matex_ksp_monitor_true_residual -ksp_max_it 10 -matex_ksp_initial_guess_nonzero -matex_ksp_max_it 1
```
  Final error=2.48070e-01

I'd change the strategy.
































  







 
























  
