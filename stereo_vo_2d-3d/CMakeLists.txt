CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

PROJECT( vo )
SET( CMAKE_CXX_FLAGS "-std=c++11")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

file(GLOB vo
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${vo} )
target_link_libraries( vo
    ${OpenCV_LIBS}
)
