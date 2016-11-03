project (neon_rpi3)
 
SET(GCC_COVERAGE_COMPILE_FLAGS "-O2 -Wall -D_REENTRANT -march=armv8-a -mcpu=cortex-a53")
SET(GCC_COVERAGE_LINK_FLAGS "-lm -lpthread")
 
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )
 
add_executable(neon_rpi3 main.cpp sampen.cpp sampen_neon.cpp)
 
