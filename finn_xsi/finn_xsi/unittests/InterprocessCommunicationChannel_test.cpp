#include "InterprocessCommunicationChannel.hpp"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <thread>

// Simple request/response types for testing
struct TestRequest {
    int value;
    bool flag;

    TestRequest() : value(0), flag(false) {}
    TestRequest(int v, bool f) : value(v), flag(f) {}

    bool operator==(const TestRequest& other) const { return value == other.value && flag == other.flag; }
};

struct TestResponse {
    int result;
    bool success;

    TestResponse() : result(0), success(false) {}
    TestResponse(int r, bool s) : result(r), success(s) {}

    bool operator==(const TestResponse& other) const { return result == other.result && success == other.success; }
};

// Test fixture for InterprocessCommunicationChannel tests
class InterprocessCommunicationChannelTest : public ::testing::Test {
     protected:
    void SetUp() override {
        // Generate unique shared memory name for each test
        shmName = "test_ipc_" + std::to_string(getpid()) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void TearDown() override {
        // Cleanup: ensure shared memory is removed
        boost::interprocess::shared_memory_object::remove(shmName.c_str());
    }

    std::string shmName;
};

// ===== Constructor and Initialization Tests =====

TEST_F(InterprocessCommunicationChannelTest, SenderConstructorCreatesSharedMemory) {
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

    // Verify that shared memory exists
    bool shmExists = false;
    try {
        boost::interprocess::managed_shared_memory shmem(boost::interprocess::open_only, shmName.c_str());
        shmExists = true;
    } catch (...) { shmExists = false; }

    EXPECT_TRUE(shmExists);
}

TEST_F(InterprocessCommunicationChannelTest, ReceiverWaitsForSenderToCreateSharedMemory) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver (waits for sender)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        exit(0);
    } else {
        // Parent process: Sender (creates shared memory)
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        // Wait for child to complete
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, DefaultConstructorCreatesUninitializedObject) {
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> channel;
    // Should not crash - object is in moved-from state
    // Destructor should handle this gracefully
}

TEST_F(InterprocessCommunicationChannelTest, MoveConstructorTransfersOwnership) {
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender1(shmName);
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender2(std::move(sender1));

    // sender2 should now own the shared memory
    // sender1 should be in moved-from state (destructor shouldn't crash)
}

TEST_F(InterprocessCommunicationChannelTest, MoveAssignmentTransfersOwnership) {
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender1(shmName);
    InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender2;

    sender2 = std::move(sender1);

    // sender2 should now own the shared memory
    // sender1 should be in moved-from state
}

// ===== Single Request-Response Tests =====

TEST_F(InterprocessCommunicationChannelTest, SingleRequestResponseExchange) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender sends request, waits for response
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        TestRequest req(42, true);
        TestResponse resp = sender.send_request(req);

        // Verify response
        exit((resp.result == 84 && resp.success) ? 0 : 1);
    } else {
        // Parent process: Receiver waits for request, sends response
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);

        // Small delay to ensure both processes are ready
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        EXPECT_EQ(req.value, 42);
        EXPECT_TRUE(req.flag);

        // Send response (double the request value)
        TestResponse resp(req.value * 2, true);
        receiver.send_response(resp);

        // Wait for child and check result
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, RequestResponseWithDifferentValues) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        TestRequest req(100, false);
        TestResponse resp = sender.send_request(req);

        exit((resp.result == 200 && !resp.success) ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        EXPECT_EQ(req.value, 100);
        EXPECT_FALSE(req.flag);

        TestResponse resp(req.value * 2, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, SingleSplitJoinRequest) {
    // Test that a diamond pattern of communication works
    pid_t p1 = fork();
    pid_t p2 = fork();
    pid_t p3 = fork();
    std::string leftName = shmName + "_left_in";
    std::string rightName = shmName + "_right_in";
    std::string leftOutName = shmName + "_left_out";
    std::string rightOutName = shmName + "_right_out";

    if (p1 != 0 && p2 != 0 && p3 != 0) {
        // Parent (origin)
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> originToLeft(leftName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> originToRight(rightName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        originToLeft.handshake();
        originToRight.handshake();

        // Send message to the left
        TestRequest reqLeft(100, false);
        TestResponse respLeft = originToLeft.send_request(reqLeft);
        EXPECT_EQ(respLeft.result, 600);

        // Send message to the right
        TestRequest reqRight(130, false);
        TestResponse respRight = originToRight.send_request(reqRight);
        EXPECT_EQ(respRight.result, 780);
        std::cout << "Origin done." << std::endl;

    } else if (p1 == 0 && p2 != 0 && p3 != 0) {
        // P1 (Left)
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> fromOrigin(leftName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> toEnd(leftOutName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        fromOrigin.handshake();
        toEnd.handshake();

        // Receive from origin
        TestRequest req = fromOrigin.receive_request();
        EXPECT_EQ(req.value, 100);
        EXPECT_FALSE(req.flag);

        // Forward triple
        TestRequest reqForward(req.value * 3, req.flag);
        TestResponse resp = toEnd.send_request(reqForward);
        auto expectedResponseFromEnd = req.value * 2 * 3;
        EXPECT_EQ(resp.result, expectedResponseFromEnd);

        // Answer with value from end
        TestResponse respOrigin(resp.result, resp.success);
        fromOrigin.send_response(respOrigin);

        std::cout << "Left done." << std::endl;
        exit((req.value == 100 && resp.result == expectedResponseFromEnd) ? 0 : 1);

    } else if (p1 != 0 && p2 == 0 && p3 != 0) {
        // P2 (Right)
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> fromOrigin(rightName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> toEnd(rightOutName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        fromOrigin.handshake();
        toEnd.handshake();

        // Receive from origin
        TestRequest req = fromOrigin.receive_request();
        EXPECT_EQ(req.value, 130);
        EXPECT_FALSE(req.flag);

        // Forward triple
        TestRequest reqForward(req.value * 3, req.flag);
        TestResponse resp = toEnd.send_request(reqForward);
        auto expectedResponseFromEnd = req.value * 2 * 3;
        EXPECT_EQ(resp.result, expectedResponseFromEnd);

        // Answer with value from end
        TestResponse respOrigin(resp.result, resp.success);
        fromOrigin.send_response(respOrigin);

        std::cout << "Right done." << std::endl;
        exit((req.value == 130 && resp.result == expectedResponseFromEnd) ? 0 : 1);

    } else if (p1 != 0 && p2 != 0 && p3 == 0) {
        // End
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> endLeft(leftOutName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> endRight(rightOutName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        endLeft.handshake();
        endRight.handshake();

        // Receive and return double
        TestRequest reqLeft = endLeft.receive_request();
        EXPECT_EQ(reqLeft.value, 300);
        TestResponse respLeft(reqLeft.value * 2, true);
        endLeft.send_response(respLeft);

        // Receive and return double
        TestRequest reqRight = endRight.receive_request();
        EXPECT_EQ(reqRight.value, 390);
        TestResponse respRight(reqRight.value * 2, true);
        endRight.send_response(respRight);

        std::cout << "End done." << std::endl;
        exit((reqLeft.value == 300 && reqRight.value == 390) ? 0 : 1);
    }

    // Wait for all forks to shut down
    if (p1 != 0 && p2 != 0 && p3 != 0) {
        int status;
        waitpid(p1, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
        waitpid(p2, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
        waitpid(p3, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }

}

// ===== Multiple Request-Response Tests =====

TEST_F(InterprocessCommunicationChannelTest, MultipleRequestResponseSequential) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender sends multiple requests
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 10; ++i) {
            TestRequest req(i, i % 2 == 0);
            TestResponse resp = sender.send_request(req);

            // Verify response matches expected calculation
            if (resp.result != i * 3 || resp.success != (i % 2 == 0)) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver processes multiple requests
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 10; ++i) {
            TestRequest req = receiver.receive_request();
            EXPECT_EQ(req.value, i);
            EXPECT_EQ(req.flag, i % 2 == 0);

            // Send calculated response
            TestResponse resp(req.value * 3, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, ManyRequestResponseExchanges) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 1000; ++i) {
            TestRequest req(i % 100, i % 3 == 0);
            TestResponse resp = sender.send_request(req);

            // Verify response
            int expected = (i % 100) + 10;
            if (resp.result != expected) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 1000; ++i) {
            TestRequest req = receiver.receive_request();

            // Just verify exchange completes without deadlock
            int expected_val = i % 100;
            EXPECT_EQ(req.value, expected_val);

            TestResponse resp(req.value + 10, true);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, AlternatingRequestPattern) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with alternating pattern
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 100; ++i) {
            bool flag = (i % 2 == 0);
            TestRequest req(i, flag);
            TestResponse resp = sender.send_request(req);

            // Verify response
            if (resp.result != i * 2 || resp.success != flag) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 100; ++i) {
            TestRequest req = receiver.receive_request();
            EXPECT_EQ(req.value, i);
            EXPECT_EQ(req.flag, i % 2 == 0);

            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Buffer Flipping Tests =====

TEST_F(InterprocessCommunicationChannelTest, BufferFlipsCorrectly) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        // Perform multiple exchanges to trigger buffer flips
        for (int i = 0; i < 20; ++i) {
            TestRequest req(i, true);
            TestResponse resp = sender.send_request(req);

            if (resp.result != i + 1) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Perform multiple exchanges - buffer should flip multiple times
        for (int i = 0; i < 20; ++i) {
            TestRequest req = receiver.receive_request();
            EXPECT_EQ(req.value, i);

            TestResponse resp(req.value + 1, true);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Stress Tests =====

TEST_F(InterprocessCommunicationChannelTest, HighFrequencyExchanges) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - rapid exchanges
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 10000; ++i) {
            TestRequest req(i & 0xFF, i & 1);
            TestResponse resp = sender.send_request(req);

            if (resp.result != (i & 0xFF) * 2) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver - rapid exchanges
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 10000; ++i) {
            TestRequest req = receiver.receive_request();

            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, StressTestWithComplexPattern) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with complex pattern
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 5000; ++i) {
            int val = (i * 7) % 127;
            bool flag = ((i * 11) % 13) < 6;
            TestRequest req(val, flag);
            TestResponse resp = sender.send_request(req);

            if (resp.result != val + 5) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver with response calculation
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 5000; ++i) {
            TestRequest req = receiver.receive_request();

            TestResponse resp(req.value + 5, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Reference Counting Tests =====

TEST_F(InterprocessCommunicationChannelTest, ReferenceCountingTwoProcesses) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Create sender and let it go out of scope
        {
            InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);
            TestRequest req(1, true);
            sender.send_request(req);
        }

        // Shared memory should still exist because parent still holds reference
        bool shmExists = false;
        try {
            boost::interprocess::managed_shared_memory shmem(boost::interprocess::open_only, shmName.c_str());
            shmExists = true;
        } catch (...) { shmExists = false; }

        exit(shmExists ? 0 : 1);
    } else {
        // Parent process: Keep receiver alive
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        TestResponse resp(req.value, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, SharedMemoryCleanupAfterBothProcessesExit) {
    // This test verifies that shared memory is properly cleaned up
    // when both processes exit.

    pid_t verifier_pid = fork();

    if (verifier_pid == 0) {
        // Verifier process: spawns two children and then checks cleanup
        pid_t sender_pid = fork();

        if (sender_pid == 0) {
            // First child: Sender
            // Use block scope so destructor is called before exit
            {
                InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);
                TestRequest req(42, true);
                sender.send_request(req);
            }  // Destructor called here
            exit(0);
        }

        // Small delay to ensure sender creates shared memory
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        pid_t receiver_pid = fork();
        if (receiver_pid == 0) {
            // Second child: Receiver
            // Use block scope so destructor is called before exit
            {
                InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
                TestRequest req = receiver.receive_request();
                TestResponse resp(req.value, req.flag);
                receiver.send_response(resp);
            }  // Destructor called here
            exit(0);
        }

        // Wait for both children to complete
        int sender_status, receiver_status;
        waitpid(sender_pid, &sender_status, 0);
        waitpid(receiver_pid, &receiver_status, 0);

        // Give time for cleanup to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Verify shared memory is cleaned up
        bool shmExists = false;
        try {
            boost::interprocess::managed_shared_memory shmem(boost::interprocess::open_only, shmName.c_str());
            shmExists = true;
        } catch (...) { shmExists = false; }

        // Exit with 0 if cleanup succeeded (shmExists == false)
        exit(shmExists ? 1 : 0);
    } else {
        // Parent: Wait for verifier process
        int status;
        waitpid(verifier_pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Move Semantics Tests =====

TEST_F(InterprocessCommunicationChannelTest, MoveConstructorMaintainsConnection) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with move
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender1(shmName);
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender2(std::move(sender1));

        TestRequest req(99, false);
        TestResponse resp = sender2.send_request(req);

        exit((resp.result == 99 && !resp.success) ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        EXPECT_EQ(req.value, 99);
        EXPECT_FALSE(req.flag);

        TestResponse resp(req.value, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, MoveAssignmentMaintainsConnection) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with move assignment
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender1(shmName);
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender2;
        sender2 = std::move(sender1);

        TestRequest req(77, true);
        TestResponse resp = sender2.send_request(req);

        exit((resp.result == 77 && resp.success) ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        EXPECT_EQ(req.value, 77);
        EXPECT_TRUE(req.flag);

        TestResponse resp(req.value, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Edge Cases =====

TEST_F(InterprocessCommunicationChannelTest, FirstCallBehavior) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - first call should not wait for buffer flip
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        auto start = std::chrono::steady_clock::now();
        TestRequest req(1, true);
        sender.send_request(req);
        auto end = std::chrono::steady_clock::now();

        // First call should complete quickly (not waiting for previous flip)
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        exit(duration.count() < 100 ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        TestResponse resp(req.value, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, ConsecutiveRequestsSameValue) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Send same request repeatedly
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 50; ++i) {
            TestRequest req(123, true);
            TestResponse resp = sender.send_request(req);

            if (resp.result != 123 || !resp.success) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Verify same request received repeatedly
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 50; ++i) {
            TestRequest req = receiver.receive_request();
            EXPECT_EQ(req.value, 123);
            EXPECT_TRUE(req.flag);

            TestResponse resp(req.value, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Timing and Synchronization Tests =====

TEST_F(InterprocessCommunicationChannelTest, SynchronizationBetweenProcesses) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - delayed start
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        for (int i = 0; i < 10; ++i) {
            TestRequest req(i, true);
            sender.send_request(req);
        }
        exit(0);
    } else {
        // Parent process: Receiver - starts immediately
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Should wait for sender to be ready
        for (int i = 0; i < 10; ++i) {
            TestRequest req = receiver.receive_request();
            EXPECT_EQ(req.value, i);

            TestResponse resp(req.value, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Custom Shared Memory Size Tests =====

TEST_F(InterprocessCommunicationChannelTest, CustomSharedMemorySize) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with larger shared memory
        InterprocessCommunicationChannel<TestRequest, TestResponse, true, 8192> sender(shmName);

        TestRequest req(55, false);
        TestResponse resp = sender.send_request(req);

        exit((resp.result == 55) ? 0 : 1);
    } else {
        // Parent process: Receiver with larger shared memory
        InterprocessCommunicationChannel<TestRequest, TestResponse, false, 8192> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TestRequest req = receiver.receive_request();
        EXPECT_EQ(req.value, 55);

        TestResponse resp(req.value, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Stop Token Tests =====

TEST_F(InterprocessCommunicationChannelTest, SenderCancellationViaStopToken) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender that gets cancelled while waiting for response
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;
        std::jthread canceller([&stop_src]() {
            // Cancel after 100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            stop_src.request_stop();
        });

        TestRequest req(42, true);
        auto start = std::chrono::steady_clock::now();
        TestResponse resp = sender.send_request(req, stop_src.get_token());
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should return default response quickly (within 200ms, accounting for scheduling)
        // and not wait indefinitely for the receiver that never responds
        exit((duration.count() < 200 && resp.result == 0 && !resp.success) ? 0 : 1);
    } else {
        // Parent process: Sender creates shared memory but receiver never responds
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> dummy_sender(shmName);

        // Wait for child to complete
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, ReceiverCancellationViaStopToken) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver that gets cancelled while waiting for request
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);

        std::stop_source stop_src;
        std::jthread canceller([&stop_src]() {
            // Cancel after 100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            stop_src.request_stop();
        });

        auto start = std::chrono::steady_clock::now();
        TestRequest req = receiver.receive_request(stop_src.get_token());
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should return default request quickly (within 200ms)
        // and not wait indefinitely for a request that never comes
        exit((duration.count() < 200 && req.value == 0 && !req.flag) ? 0 : 1);
    } else {
        // Parent process: Sender creates shared memory but never sends request
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        // Wait for child to complete
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, StopTokenDoesNotInterruptNormalOperation) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with stop token that is never triggered
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;

        TestRequest req(99, true);
        TestResponse resp = sender.send_request(req, stop_src.get_token());

        // Should complete normally and receive proper response
        exit((resp.result == 198 && resp.success) ? 0 : 1);
    } else {
        // Parent process: Receiver responds normally
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::stop_source stop_src;
        TestRequest req = receiver.receive_request(stop_src.get_token());
        EXPECT_EQ(req.value, 99);
        EXPECT_TRUE(req.flag);

        TestResponse resp(req.value * 2, req.flag);
        receiver.send_response(resp);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, MultipleExchangesWithStopToken) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender performs multiple exchanges with stop token
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;

        for (int i = 0; i < 50; ++i) {
            TestRequest req(i, i % 2 == 0);
            TestResponse resp = sender.send_request(req, stop_src.get_token());

            if (resp.result != i * 2 || resp.success != (i % 2 == 0)) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver with stop token
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::stop_source stop_src;

        for (int i = 0; i < 50; ++i) {
            TestRequest req = receiver.receive_request(stop_src.get_token());
            EXPECT_EQ(req.value, i);

            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, SenderCancellationMidExchange) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender that gets cancelled after some exchanges
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;

        // Perform a few successful exchanges
        for (int i = 0; i < 5; ++i) {
            TestRequest req(i, true);
            TestResponse resp = sender.send_request(req, stop_src.get_token());

            if (resp.result != i * 2) {
                exit(1);
            }
        }

        // Now trigger cancellation for next exchange
        std::jthread canceller([&stop_src]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            stop_src.request_stop();
        });

        // This should be cancelled (receiver won't respond in time)
        TestRequest req(100, false);
        auto start = std::chrono::steady_clock::now();
        TestResponse resp = sender.send_request(req, stop_src.get_token());
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should return default response due to cancellation
        exit((duration.count() < 200 && resp.result == 0) ? 0 : 1);
    } else {
        // Parent process: Receiver responds to first 5 requests, then delays
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::stop_source stop_src;

        // Respond to first 5 requests normally
        for (int i = 0; i < 5; ++i) {
            TestRequest req = receiver.receive_request(stop_src.get_token());
            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        // Delay before processing the 6th request (which will be cancelled)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Try to receive next request (might be cancelled)
        TestRequest req = receiver.receive_request(stop_src.get_token());
        if (req.value == 100) {
            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, ReceiverCancellationMidExchange) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver that gets cancelled after some exchanges
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);

        std::stop_source stop_src;

        // Perform a few successful exchanges
        for (int i = 0; i < 5; ++i) {
            TestRequest req = receiver.receive_request(stop_src.get_token());

            if (req.value != i) {
                exit(1);
            }

            TestResponse resp(req.value * 2, req.flag);
            receiver.send_response(resp);
        }

        // Now trigger cancellation for next receive
        std::jthread canceller([&stop_src]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            stop_src.request_stop();
        });

        // This should be cancelled (sender will delay)
        auto start = std::chrono::steady_clock::now();
        TestRequest req = receiver.receive_request(stop_src.get_token());
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should return default request due to cancellation
        exit((duration.count() < 200 && req.value == 0) ? 0 : 1);
    } else {
        // Parent process: Sender sends first 5 requests, then delays
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::stop_source stop_src;

        // Send first 5 requests normally
        for (int i = 0; i < 5; ++i) {
            TestRequest req(i, true);
            TestResponse resp = sender.send_request(req, stop_src.get_token());

            if (resp.result != i * 2) {
                // Unexpected response
                break;
            }
        }

        // Delay before sending the 6th request (receiver will be cancelled)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, ImmediateCancellation) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with immediately stopped token
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;
        stop_src.request_stop();  // Stop immediately

        TestRequest req(42, true);
        auto start = std::chrono::steady_clock::now();
        TestResponse resp = sender.send_request(req, stop_src.get_token());
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should return immediately with default response
        exit((duration.count() < 50 && resp.result == 0 && !resp.success) ? 0 : 1);
    } else {
        // Parent process: Just creates sender
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> dummy_sender(shmName);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterprocessCommunicationChannelTest, StopTokenWithHighFrequencyExchanges) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with many rapid exchanges using stop token
        InterprocessCommunicationChannel<TestRequest, TestResponse, true> sender(shmName);

        std::stop_source stop_src;

        for (int i = 0; i < 100; ++i) {
            TestRequest req(i % 10, i % 2 == 0);
            TestResponse resp = sender.send_request(req, stop_src.get_token());

            if (resp.result != (i % 10) * 3) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver with stop token
        InterprocessCommunicationChannel<TestRequest, TestResponse, false> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::stop_source stop_src;

        for (int i = 0; i < 100; ++i) {
            TestRequest req = receiver.receive_request(stop_src.get_token());

            TestResponse resp(req.value * 3, req.flag);
            receiver.send_response(resp);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
