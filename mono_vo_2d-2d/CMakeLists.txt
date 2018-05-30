cmake_minimum_required(VERSION 2.8)
project( vo )

find_package( OpenCV 3.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB vo
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${vo} )
target_link_libraries( vo ${OpenCV_LIBS} )
