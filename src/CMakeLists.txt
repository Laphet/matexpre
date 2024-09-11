add_executable(main test_petsc_dmda.cxx)

# Manually set the include and library directories
target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(main PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(main PUBLIC ${SLEPC_LIBRARIES} GSL::gsl hdf5::hdf5 PkgConfig::FFTW3)

