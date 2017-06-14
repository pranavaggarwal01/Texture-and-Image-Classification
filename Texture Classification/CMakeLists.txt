cmake_minimum_required(VERSION 2.8)
project( Homework3_1 )
find_package( OpenCV REQUIRED )
add_executable( P1a Problem1_a.cpp )
add_executable( P1b Problem1_b.cpp )
target_link_libraries( P1a  ${OpenCV_LIBS} )
target_link_libraries( P1b  ${OpenCV_LIBS} )


