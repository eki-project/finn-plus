# Enable testing
enable_testing()

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add FIFO unit tests
add_executable(FIFO_test FIFO_test.cpp ${CORE_SRC})
target_link_libraries(FIFO_test PRIVATE nlohmann_json::nlohmann_json GTest::gtest_main Threads::Threads -ldl -lrt)
target_include_directories(FIFO_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(FIFO_test PUBLIC "$ENV{XILINX_VIVADO}/data/xsim/include")

# Add InterprocessCommunicationChannel unit tests
add_executable(InterprocessCommunicationChannel_test InterprocessCommunicationChannel_test.cpp)
target_link_libraries(InterprocessCommunicationChannel_test PRIVATE GTest::gtest_main Threads::Threads -ldl -lrt nlohmann_json::nlohmann_json)
target_include_directories(InterprocessCommunicationChannel_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${Boost_INCLUDE_DIRS})

# Add Integration tests (FIFO + InterSimulationInterface)
add_executable(Integration_test Integration_test.cpp ${CORE_SRC})
target_link_libraries(Integration_test PRIVATE nlohmann_json::nlohmann_json GTest::gtest_main Threads::Threads -ldl -lrt)
target_include_directories(Integration_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${Boost_INCLUDE_DIRS})
target_include_directories(Integration_test PUBLIC "$ENV{XILINX_VIVADO}/data/xsim/include")

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(FIFO_test)
gtest_discover_tests(InterprocessCommunicationChannel_test)
gtest_discover_tests(Integration_test)

# Create a target to build all unittests at once
add_custom_target(all_unittests)
add_dependencies(all_unittests FIFO_test InterprocessCommunicationChannel_test Integration_test)
