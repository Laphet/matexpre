add_executable(main 
solver.cxx 
# test_matex.cxx
test_marmousi.cxx
)

# Manually set the include and library directories for SLEPc.
target_include_directories(main PUBLIC ${SLEPC_INCLUDE_DIRS})

target_link_directories(main PUBLIC ${SLEPC_LIBRARY_DIRS})

target_link_libraries(main PUBLIC ${SLEPC_LINK_LIBRARIES})

