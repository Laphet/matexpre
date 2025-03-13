add_executable(main 
solver.cxx 
test_matex.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(main PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(main PUBLIC ${SLEPC_LINK_LIBRARIES})

add_executable(marmousi_c4 
solver.cxx 
test_marmousi_c4.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(marmousi_c4 PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(marmousi_c4 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(marmousi_c4 PUBLIC ${SLEPC_LINK_LIBRARIES})

add_executable(marmousi_c2 
solver.cxx 
test_marmousi_c2.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(marmousi_c2 PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(marmousi_c2 PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(marmousi_c2 PUBLIC ${SLEPC_LINK_LIBRARIES})

add_executable(conv_lens 
solver.cxx 
test_conv_lens.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(conv_lens PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(conv_lens PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(conv_lens PUBLIC ${SLEPC_LINK_LIBRARIES})

add_executable(constant_3d
solver.cxx
test_constant_3d.cxx
)
# Manually set the include and library directories for SLEPc.
target_include_directories(constant_3d PUBLIC ${SLEPC_INCLUDE_DIRS})
target_link_directories(constant_3d PUBLIC ${SLEPC_LIBRARY_DIRS})
target_link_libraries(constant_3d PUBLIC ${SLEPC_LINK_LIBRARIES})

