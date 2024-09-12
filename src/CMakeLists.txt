add_executable(main get_mat_eigenvals.cxx matexpre.cxx hello_hdf5.cxx)

# Manually set the include and library directories for SLEPc and FFTW3.
target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS})

target_link_directories(main PUBLIC ${FFTW3_LIBRARY_DIRS})
# message("SLEPC_LIBRARIES: ${SLEPC_LIBRARIES}")
# message("SLEPC_LINK_LIBRARIES: ${SLEPC_LINK_LIBRARIES}")

# Due to errors in FFTW3.pc, which does not include -lfftw3_mpi, we need to add it manually.
list(PREPEND FFTW3_LIBRARIES "fftw3_mpi")
# message("FFTW3_LIBRARIES: ${FFTW3_LIBRARIES}")

target_link_libraries(main PUBLIC ${SLEPC_LINK_LIBRARIES} ${FFTW3_LIBRARIES} GSL::gsl hdf5::hdf5)

