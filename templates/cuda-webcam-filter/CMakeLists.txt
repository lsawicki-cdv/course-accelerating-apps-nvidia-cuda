cmake_minimum_required(VERSION 3.27) # Requires 25 or above to have CUDA20 standard flag

if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(COMPILER_CACHE_FILE "${CMAKE_BINARY_DIR}/compiler_settings.cmake")

    if(WIN32)
        # Windows compiler detection
        find_program(MSVC_COMPILER cl)
        if(MSVC_COMPILER)
            file(WRITE "${COMPILER_CACHE_FILE}"
                "set(CMAKE_C_COMPILER \"${MSVC_COMPILER}\" CACHE FILEPATH \"C compiler\" FORCE)\n"
                "set(CMAKE_CXX_COMPILER \"${MSVC_COMPILER}\" CACHE FILEPATH \"C++ compiler\" FORCE)\n"
            )
            set(CMAKE_C_COMPILER "${MSVC_COMPILER}" CACHE FILEPATH "C compiler" FORCE)
            set(CMAKE_CXX_COMPILER "${MSVC_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
        endif()
    else()
        # Only try to detect compilers if they haven't been specified
        find_program(GCC_12_COMPILER gcc-12)
        find_program(GXX_12_COMPILER g++-12)
        
        if(GCC_12_COMPILER AND GXX_12_COMPILER)
            # Generate a cache initialization file
            file(WRITE "${COMPILER_CACHE_FILE}"
                "set(CMAKE_C_COMPILER \"${GCC_12_COMPILER}\" CACHE FILEPATH \"C compiler\" FORCE)\n"
                "set(CMAKE_CXX_COMPILER \"${GXX_12_COMPILER}\" CACHE FILEPATH \"C++ compiler\" FORCE)\n"
                "set(CMAKE_CUDA_HOST_COMPILER \"${GXX_12_COMPILER}\" CACHE FILEPATH \"CUDA host compiler\" FORCE)\n"
            )
            
            # Output message about using the file next time
            message(STATUS "Created compiler settings cache file. For faster configuration, use:")
            message(STATUS "  cmake -C ${COMPILER_CACHE_FILE} ..")
            
            # Apply the settings right now
            set(CMAKE_C_COMPILER "${GCC_12_COMPILER}" CACHE FILEPATH "C compiler" FORCE)
            set(CMAKE_CXX_COMPILER "${GXX_12_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
            # CUDA host compiler will be set after CUDA language is enabled
        endif()
    endif()
endif()

project(cuda-webcam-filter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# Set build type to Release for better performance
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

# Project version
set(PROJECT_VERSION "0.1.0")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/version.h
    @ONLY
)

# Options
option(CUDA_SUPPORT "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA debug" OFF)
option(RUN_UNIT_TESTS "Enable running unit tests" OFF)
option(RUN_FUNCTIONAL_TESTS "Enable running functional tests" OFF)

# Bundled dependencies
add_subdirectory(external/cxxopts)
add_subdirectory(external/plog)
add_subdirectory(external/gtest)

include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)

# OS-specific configurations
if(WIN32)
    # Windows-specific settings
    message(STATUS "Configuring build for Windows")

    # Windows-specific CUDA paths
    if(CUDA_SUPPORT)
        # Try to find CUDA in standard locations
        find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
            HINTS 
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"
            PATH_SUFFIXES cuda
        )
        if(CUDA_INCLUDE_DIRS)
            include_directories(${CUDA_INCLUDE_DIRS})
            message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")
        endif()
    endif()
else()
    # Linux-specific settings
    message(STATUS "Configuring build for Linux")

    if(GCC_12_COMPILER AND GXX_12_COMPILER)
        message(STATUS "Using GCC 12: ${GCC_12_COMPILER}")
        message(STATUS "Using G++ 12: ${GXX_12_COMPILER}")
    
        # Only set CUDA host compiler here if CUDA is enabled separately
        if(CUDA_SUPPORT)
            set(CMAKE_CUDA_HOST_COMPILER "${GXX_12_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
        endif()
    else()
        message(WARNING "GCC 12 not found. Please install it with: sudo apt-get install gcc-12 g++-12")
    endif()
    
    # Linux-specific CUDA paths
    if(CUDA_SUPPORT)
    find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
            HINTS /usr/local/cuda/include /usr/include
            PATH_SUFFIXES cuda
        )
        if(CUDA_INCLUDE_DIRS)
            include_directories(${CUDA_INCLUDE_DIRS})
            message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")
        endif()
    endif()
endif()

# CUDA support
if(CUDA_SUPPORT)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

    # Include CUDA toolkit directories
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    endif()

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++20")

    if(CUDA_DEBUG)
        message(STATUS "CUDA Debug is ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") # enable cuda-gdb
    endif()

    message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")

    # Target architecture
    set(CMAKE_CUDA_ARCHITECTURES "89")  # TODO: RTX 4060 (Ada Lovelace) - Adjust to target specific GPUs!!!
endif()

# Find OpenCV (must be installed system-wide or via CMAKE_PREFIX_PATH)
# See README.md for installation instructions on Linux and Windows
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
message(STATUS "OpenCV ${OpenCV_VERSION} found")
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/input_handler.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/filter_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/input_args_parser/input_args_parser.cpp
)

# CUDA specific files
if(CUDA_SUPPORT)
    list(APPEND SOURCES
        ${CMAKE_SOURCE_DIR}/src/kernels/convolution_kernels.cu
    )
    # Treat main.cpp as CUDA file
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/main.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/kernels/
    ${CMAKE_SOURCE_DIR}/src/utils/
    ${CMAKE_SOURCE_DIR}/src/input_args_parser/
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    plog::plog
    GTest::gtest GTest::gmock
)

# Testing
if(RUN_UNIT_TESTS)
    add_subdirectory(tests/unit_tests)
endif()

# Not implemented
# if(RUN_FUNCTIONAL_TESTS)
#     add_subdirectory(tests/functional_tests)
# endif()

if((NOT RUN_UNIT_TESTS) AND (NOT RUN_FUNCTIONAL_TESTS))
    add_custom_target(run_tests
        COMMAND echo "Running tests skipped...")
endif()