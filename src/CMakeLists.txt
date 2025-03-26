add_library(matexpre solver.cxx)
target_include_directories(matexpre PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(matexpre PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(matexpre PUBLIC ${SLEPC_LINK_LIBRARIES})


add_executable(main 
test_matex.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(main PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(main PUBLIC matexpre)

add_executable(marmousi_c4 
test_marmousi_c4.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(marmousi_c4 PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(marmousi_c4 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(marmousi_c4 PUBLIC matexpre)

add_executable(marmousi_c2 
# solver.cxx 
test_marmousi_c2.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(marmousi_c2 PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(marmousi_c2 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(marmousi_c2 PUBLIC matexpre)

add_executable(conv_lens 
# solver.cxx 
test_conv_lens.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(conv_lens PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(conv_lens PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(conv_lens PUBLIC matexpre)

add_executable(constant_3d
# solver.cxx
test_constant_3d.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(constant_3d PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(constant_3d PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(constant_3d PUBLIC matexpre)

add_executable(salt_dome
# solver.cxx
test_salt_dome.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(salt_dome PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(salt_dome PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(salt_dome PUBLIC matexpre)

add_executable(overthrust
# solver.cxx
test_overthrust.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(overthrust PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(overthrust PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(overthrust PUBLIC matexpre)

add_executable(conv_lens_3d 
# solver.cxx 
test_conv_lens_3d.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(conv_lens_3d PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(conv_lens_3d PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(conv_lens_3d PUBLIC matexpre)

add_executable(overthrust_c2
# solver.cxx
test_overthrust_c2.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(overthrust_c2 PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(overthrust_c2 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(overthrust_c2 PUBLIC matexpre)

add_executable(salt_dome_c2
# solver.cxx
test_salt_dome_c2.cxx
)
# Manually set the include and library directories for SLEPc.
# target_include_directories(salt_dome_c2 PUBLIC ${SLEPC_INCLUDE_DIRS})
# target_link_directories(salt_dome_c2 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(salt_dome_c2 PUBLIC matexpre)

add_executable(test_hdf5
test_hdf5.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(test_hdf5 PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(test_hdf5 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(test_hdf5 PUBLIC ${SLEPC_LINK_LIBRARIES})