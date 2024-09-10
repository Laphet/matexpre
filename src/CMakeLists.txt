add_executable(main get_mat_eigenvals.cxx matexpre.cxx hello_hdf5.cxx)

# Manually set the include and library directories
target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(main PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(main PUBLIC ${SLEPC_LIBRARIES} GSL::gsl hdf5::hdf5 PkgConfig::FFTW3)

