message(STATUS "Configuring OpenCV")

set(OPENCV_VERSION 4.6.0)
set(
        OPENCV_PUBLIC_LIBRARIES
        opencv_core opencv_calib3d opencv_imgproc opencv_imgcodecs opencv_features2d opencv_highgui opencv_xfeatures2d
)
# set(OPENCV_PRIVATE_LIBRARIES  )               TODO: research which modules we don't use

get_filename_component(OPENCV_BUILD_DIRS_ROOT ../../build REALPATH)
file(GLOB OPENCV_BUILD_DIRS ${OPENCV_BUILD_DIRS_ROOT}/opencv-build*)
list(APPEND OpenCV_ROOT ${OPENCV_BUILD_DIRS})

# Android Gradle Plugin fails to find OpenCV from OpenCV_ROOT automatically for some reason
if (ANDROID)
    foreach (OPENCV_BUILD_DIR ${OPENCV_BUILD_DIRS})
        message(STATUS "Manually considering OpenCV directory: ${OPENCV_BUILD_DIR}")
        set(OpenCV_DIR ${OPENCV_BUILD_DIR})
        find_package(OpenCV ${OPENCV_VERSION} COMPONENTS ${OPENCV_PUBLIC_LIBRARIES} ${OPENCV_PRIVATE_LIBRARIES})
        if (OpenCV_FOUND)
            break()
        endif ()
    endforeach ()
endif ()

find_package(OpenCV ${OPENCV_VERSION} REQUIRED ${OPENCV_PUBLIC_LIBRARIES} ${OPENCV_PRIVATE_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENCV_PUBLIC_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENCV_PRIVATE_LIBRARIES})
message(STATUS "Configuring OpenCV - done")
