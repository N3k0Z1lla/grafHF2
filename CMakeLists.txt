cmake_minimum_required (VERSION 3.0)
project (UFO_Hami)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable (main raytrace.cpp framework.cpp)
add_compile_options (-Werror=all -Wextra -Wpedantic -Ofast -ftree-parallelize-loops)

target_link_libraries (main GL glut GLEW)
