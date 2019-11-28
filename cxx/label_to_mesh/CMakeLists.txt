cmake_minimum_required(VERSION 2.8)

project(LabelToMesh)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(${PROJECT_NAME} main.cxx )

target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES})