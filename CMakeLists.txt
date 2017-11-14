cmake_minimum_required(VERSION 3.7)

project (neon_rpi3)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

if(${ARCHITECTURE} STREQUAL "aarch64")
	SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -O3 -Wall -D_REENTRANT -mcpu=cortex-a53")
else()
	SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -O3 -Wall -D_REENTRANT -mcpu=cortex-a53 -mfpu=neon-vfpv4")
endif()

SET(GCC_COVERAGE_LINK_FLAGS "-lm -lpthread")
 
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )
 
add_executable(neon_rpi3 main.cpp sampen.cpp sampen_neon.cpp)
 
