cmake_minimum_required(VERSION 3.14)
project(MyProject LANGUAGES C CXX)

# Export compile commands for vscode, vim, zed etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Require C++11 for the example app
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All ai-coustics CMake configuration settings
set(AIC_SDK_ALLOW_DOWNLOAD ON CACHE BOOL "Allow download ai-coustics C SDK")
set(AIC_SDK_USE_STATIC ON CACHE BOOL "Use static linking for AIC SDK")
set(AIC_SDK_ROOT "" CACHE PATH "Path to local ai-coustics C SDK (always skips download if set)")
set(AIC_SDK_LICENSE "eyJ1c2VyIjoidXNlcl8zRHQxRVhrUWF1cE1rdzV5a2JLSXJEQlF5MXYiLCJ2ZXJzaW9uIjoiMiIsImFwaV9rZXkiOiI2OWViYTM1NzEwOWQ4YTE1OTliYjc5ZWE2ZTQ4OTBlMzliNTk1Mjg3ZDA2NWUxNGQyZmM1NWY3MWY1MTUxZTYzIn0=.nWHH0yLbnA4qlRtvrwFzj5HMqx+I9Rt1srSWDwE3uXj2Ax6u6gvlUbY+nHeCeWakHwSFUOH4BSr8VBAguuSppg7g0ycsmBD+H2pr7ad7xrirKVjXTr7vwPPRmGCSt4W1fMjDfqjAaaLdArOMA1tSd6e4uajtjGiOikHh0pznxtpOPAwSwmbkunz+8oRzoKZ3fjtmcPCJxdYQWj5UYy7frNCksTwIcO/5YuGRlbled0DtUN02ITgNPBNCTciU2MVGVx1XHk+wkgJ2rimtXVU3TABDbZT/Z0N4DkDc6NIoiVRnmlK1FlyEjMK/YoJGdgGpoU1qxYpFxMvm20A74M+25Q==" CACHE STRING "AIC SDK License Key")

# Path to local C++ wrapper
set(AIC_SDK_CPP_PATH "../" CACHE PATH "Path to local aic-sdk-cpp directory")

if(AIC_SDK_CPP_PATH AND EXISTS "${AIC_SDK_CPP_PATH}")
    # Use the local C++ wrapper
    add_subdirectory("${AIC_SDK_CPP_PATH}" aic_sdk)
else()
    include(FetchContent)
    # Download the C++ wrapper
    FetchContent_Declare(
        aic_sdk
        GIT_REPOSITORY https://github.com/ai-coustics/aic-sdk-cpp.git
        GIT_TAG        0.17.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(aic_sdk)
endif()

add_executable(my_app main.cpp ../src/audio_wave.c)

# Link against the SDK
target_link_libraries(my_app PRIVATE aic-sdk)

# Make audio_wave.h available
target_include_directories(my_app PRIVATE ../src)

# Pass license key as compile definition
if(AIC_SDK_LICENSE)
    target_compile_definitions(my_app PRIVATE AIC_SDK_LICENSE="${AIC_SDK_LICENSE}")
endif()

if(APPLE)
    target_link_libraries(my_app PRIVATE "-framework Security" "-framework CoreFoundation")
endif()

if(WIN32)
    target_link_libraries(my_app PRIVATE crypt32)
endif()
