CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()
SET(Boost_DEBUG 1)
SET(BOOST_ROOT "/software/boost/1.65.0/b2/")
SET(BOOST_INCLUDEDIR "/software/boost/1.65.0/b2/include/boost")
SET(BOOST_LIBRARYDIR "/software/boost/1.65.0/b2/lib")
SET(PYTHON_INCLUDE_DIR "/software/python3/3.5.2/include/")
SET(PYTHON_LIBRARY "/software/python3/3.5.2/lib/libpython3.so")

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/software/boost/1.65.0/b2/")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/software/boost/1.65.0/b2/lib")

#option(Boost_NO_SYSTEM_PATHS ON)
FIND_PACKAGE(Boost 1.65 EXACT REQUIRED COMPONENTS python-py35 REQUIRED)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
FIND_PACKAGE(PythonLibs 3.5 REQUIRED)


IF(Boost_FOUND)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} "/software/python3/3.5.2/include/" "/software/boost/1.65.0/b2/include/boost/python" "/software/boost/1.65.0/b2/include/boost" )
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  #FIND_PACKAGE(Boost 1.65.0 COMPONENTS python-py35 REQUIRED)

  ADD_LIBRARY(gibbs SHARED gibbs_py.cxx gibbs_py.cpp)
  SET_TARGET_PROPERTIES(gibbs PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR} -I/software/boost/1.65.0/b2/include/boost/python -O3 -funroll-loops" LINK_FLAGS "-I${PROJECT_SOURCE_DIR}  -I/software/boost/1.65.0/b2/include/boost/")#
  TARGET_LINK_LIBRARIES(gibbs ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})


ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()




IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
