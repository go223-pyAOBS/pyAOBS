cmake_minimum_required(VERSION 3.16)
project(tomo2d_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TOMO2D_ENABLE_OPENMP "Enable OpenMP acceleration for ray tracing" ON)
option(TOMO2D_ENABLE_NATIVE "Enable -march=native on non-MSVC compilers" OFF)
option(TOMO2D_ENABLE_FAST_MATH "Enable -ffast-math on non-MSVC compilers" OFF)

set(TOMO2D_CORE_SOURCES
  bend.cc
  bend_brent.cc
  bend_mnbrak.cc
  betaspline.cc
  corrlen.cc
  error.cc
  geom.cc
  graph.cc
  interface.cc
  inverse.cc
  jgrav.cc
  lsqr.cc
  smesh.cc
  sparse_rect.cc
  syngen.cc
  traveltime.cc
  util.cc
  zeltform.cc
  d_realft.c
  d_choldc.c
  d_four1.c
  d_jacobi.c
  nrutil.c
)

# 历史代码里这些 .c 由 C++ 调用且头文件未做 extern "C" 包装；
# 需按 C++ 编译以保持与原 g++ 构建的符号一致性。
set_source_files_properties(
  d_realft.c
  d_choldc.c
  d_four1.c
  d_jacobi.c
  nrutil.c
  PROPERTIES LANGUAGE CXX
)

add_library(tomo2d_core STATIC ${TOMO2D_CORE_SOURCES})
target_include_directories(tomo2d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  target_compile_options(tomo2d_core PRIVATE
    $<$<CONFIG:Release>:/O2>
  )
else()
  target_compile_options(tomo2d_core PRIVATE
    $<$<CONFIG:Release>:-O3>
  )
  if(TOMO2D_ENABLE_NATIVE)
    target_compile_options(tomo2d_core PRIVATE -march=native)
  endif()
  if(TOMO2D_ENABLE_FAST_MATH)
    target_compile_options(tomo2d_core PRIVATE -ffast-math)
  endif()
endif()

if(TOMO2D_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tomo2d_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for tomo2d_core")
  else()
    message(WARNING "OpenMP requested but not found. Build will proceed without OpenMP.")
  endif()
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(tomo2d_core PUBLIC m)
endif()

function(add_tomo2d_tool tool_name tool_src)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE tomo2d_core)
endfunction()

add_tomo2d_tool(gen_smesh gen_smesh.cc)
add_tomo2d_tool(tt_forward tt_forward.cc)
add_tomo2d_tool(tt_inverse tt_inverse.cc)
add_tomo2d_tool(gen_damp gen_damp.cc)
add_tomo2d_tool(gen_vcorr gen_vcorr.cc)
add_tomo2d_tool(stat_smesh stat_smesh.cc)
add_tomo2d_tool(edit_smesh edit_smesh.cc)
add_tomo2d_tool(edit_smesh_HHB edit_smesh_HHB.cc)
add_tomo2d_tool(edit_smesh_HHB2 edit_smesh_HHB2.cc)
