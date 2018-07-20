cmake_minimum_required(VERSION 2.8)
project( contours )
project( word )
project( threshold )
project ( square )
project ( main )
project ( hsv )
project ( red )

find_package( OpenCV REQUIRED )

add_executable( main main.cpp )
add_executable( contours contours.cpp )
add_executable( word word.cpp )
add_executable( threshold threshold.cpp )
add_executable (square square.cpp)
add_executable (hsv hsv.cpp)
add_executable (red red.cpp)

target_link_libraries( contours ${OpenCV_LIBS} )
target_link_libraries( word ${OpenCV_LIBS} )
target_link_libraries( threshold ${OpenCV_LIBS}) 
target_link_libraries(square ${OpenCV_LIBS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(hsv ${OpenCV_LIBS})
target_link_libraries(red ${OpenCV_LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Tesseract OCR
if(COMMAND pkg_check_modules)
  pkg_check_modules(Tesseract tesseract lept)
endif()
if(NOT Tesseract_FOUND)
  find_path(Tesseract_INCLUDE_DIR tesseract/baseapi.h
    HINTS
    /usr/include
    /usr/local/include)

  find_library(Tesseract_LIBRARY NAMES tesseract
    HINTS
    /usr/lib
    /usr/local/lib)

  find_library(Lept_LIBRARY NAMES lept
    HINTS
    /usr/lib
    /usr/local/lib)

  if(Tesseract_INCLUDE_DIR AND Tesseract_LIBRARY AND Lept_LIBRARY)
    set(Tesseract_INCLUDE_DIRS ${Tesseract_INCLUDE_DIR})
    set(Tesseract_LIBRARIES ${Tesseract_LIBRARY} ${Lept_LIBRARY})
    set(Tesseract_FOUND 1)
  endif()
endif()

