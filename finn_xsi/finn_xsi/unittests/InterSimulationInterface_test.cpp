#include "InterSimulationInterface.hpp"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <thread>

// Test fixture for InterSimulationInterface tests
class InterSimulationInterfaceTest : public ::testing::Test {
     protected:
    void SetUp() override {
        // Generate unique shared memory name for each test
        shmName = "test_shm_" + std::to_string(getpid()) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void TearDown() override {
        // Cleanup: ensure shared memory is removed
        boost::interprocess::shared_memory_object::remove(shmName.c_str());
    }

    std::string shmName;
};

// ===== Constructor and Initialization Tests =====

TEST_F(InterSimulationInterfaceTest, ReceiverConstructorCreatesSharedMemory) {
    InterSimulationInterface<true> receiver(shmName);

    // Verify that shared memory exists
    bool shmExists = false;
    try {
        boost::interprocess::managed_shared_memory shmem(boost::interprocess::open_only, shmName.c_str());
        shmExists = true;
    } catch (...) { shmExists = false; }

    EXPECT_TRUE(shmExists);
}

TEST_F(InterSimulationInterfaceTest, SenderWaitsForReceiverToCreateSharedMemory) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender (waits for receiver)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        InterSimulationInterface<false> sender(shmName);
        exit(0);
    } else {
        // Parent process: Receiver (creates shared memory)
        InterSimulationInterface<true> receiver(shmName);

        // Wait for child to complete
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, DefaultConstructorCreatesUninitializedObject) {
    InterSimulationInterface<true> interface;
    // Should not crash - object is in moved-from state
    // Destructor should handle this gracefully
}

TEST_F(InterSimulationInterfaceTest, MoveConstructorTransfersOwnership) {
    InterSimulationInterface<true> receiver1(shmName);
    InterSimulationInterface<true> receiver2(std::move(receiver1));

    // receiver2 should now own the shared memory
    // receiver1 should be in moved-from state (destructor shouldn't crash)
}

TEST_F(InterSimulationInterfaceTest, MoveAssignmentTransfersOwnership) {
    InterSimulationInterface<true> receiver1(shmName);
    InterSimulationInterface<true> receiver2;

    receiver2 = std::move(receiver1);

    // receiver2 should now own the shared memory
    // receiver1 should be in moved-from state
}

// ===== Single Exchange Tests =====

TEST_F(InterSimulationInterfaceTest, SingleExchangeBothProcesses) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);
        bool received = sender.exchange(true);

        // Sender sends true, should receive false from receiver
        exit(received ? 1 : 0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);

        // Small delay to ensure both processes are ready
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(false);

        // Receiver sends false, should receive true from sender
        EXPECT_TRUE(received);

        // Wait for child and check result
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, ExchangeBothSendTrue) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);
        bool received = sender.exchange(true);
        exit(received ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(true);
        EXPECT_TRUE(received);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, ExchangeBothSendFalse) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);
        bool received = sender.exchange(false);
        exit(received ? 1 : 0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(false);
        EXPECT_FALSE(received);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Multiple Exchange Tests =====

TEST_F(InterSimulationInterfaceTest, MultipleExchangesSequential) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 10; ++i) {
            bool send_val = (i % 2 == 0);
            bool received = sender.exchange(send_val);

            // Sender alternates true/false, receiver sends opposite
            bool expected = !send_val;
            if (received != expected) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 10; ++i) {
            bool send_val = (i % 2 != 0);  // Opposite of sender
            bool received = receiver.exchange(send_val);

            bool expected = !send_val;
            EXPECT_EQ(received, expected);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, ManyExchanges) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 1000; ++i) {
            bool send_val = (i % 3 == 0);
            sender.exchange(send_val);
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 1000; ++i) {
            bool send_val = (i % 5 == 0);
            bool received = receiver.exchange(send_val);

            // Just verify exchange completes without deadlock
            bool expected_from_sender = (i % 3 == 0);
            EXPECT_EQ(received, expected_from_sender);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, AlternatingPattern) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender sends alternating true/false
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 100; ++i) {
            bool send_val = (i % 2 == 0);
            bool received = sender.exchange(send_val);

            // Receiver also alternates, but starts with false
            bool expected = (i % 2 != 0);
            if (received != expected) {
                exit(1);
            }
        }
        exit(0);
    } else {
        // Parent process: Receiver sends alternating false/true
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 100; ++i) {
            bool send_val = (i % 2 != 0);
            bool received = receiver.exchange(send_val);

            bool expected = (i % 2 == 0);
            EXPECT_EQ(received, expected);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Buffer Flipping Tests =====

TEST_F(InterSimulationInterfaceTest, BufferFlipsCorrectly) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender
        InterSimulationInterface<false> sender(shmName);

        // Perform multiple exchanges to trigger buffer flips
        for (int i = 0; i < 20; ++i) {
            sender.exchange(true);
        }
        exit(0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Perform multiple exchanges - buffer should flip multiple times
        for (int i = 0; i < 20; ++i) {
            bool received = receiver.exchange(false);
            EXPECT_TRUE(received);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Stress Tests =====

TEST_F(InterSimulationInterfaceTest, HighFrequencyExchanges) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - rapid exchanges
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 10000; ++i) {
            sender.exchange(i & 1);  // Alternate between true/false
        }
        exit(0);
    } else {
        // Parent process: Receiver - rapid exchanges
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 10000; ++i) {
            receiver.exchange(!(i & 1));  // Opposite pattern
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, StressTestWithComplexPattern) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with complex pattern
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 5000; ++i) {
            bool val = ((i * 7) % 11) < 5;  // Pseudo-random pattern
            sender.exchange(val);
        }
        exit(0);
    } else {
        // Parent process: Receiver with different complex pattern
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 5000; ++i) {
            bool val = ((i * 13) % 17) < 8;  // Different pseudo-random pattern
            receiver.exchange(val);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Reference Counting Tests =====

TEST_F(InterSimulationInterfaceTest, ReferenceCountingTwoProcesses) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Create sender and let it go out of scope
        {
            InterSimulationInterface<false> sender(shmName);
            sender.exchange(true);
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
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        receiver.exchange(false);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, SharedMemoryCleanupAfterBothProcessesExit) {
    // This test verifies that shared memory is properly cleaned up
    // when both processes exit. We need to test from a third process
    // that was never part of the shared memory to avoid race conditions.

    pid_t verifier_pid = fork();

    if (verifier_pid == 0) {
        // Verifier process: spawns two children and then checks cleanup
        pid_t receiver_pid = fork();

        if (receiver_pid == 0) {
            // First child: Receiver
            // Use block scope so destructor is called before exit
            {
                InterSimulationInterface<true> receiver(shmName);
                receiver.exchange(true);
            }  // Destructor called here
            exit(0);
        }

        // Small delay to ensure receiver creates shared memory
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        pid_t sender_pid = fork();
        if (sender_pid == 0) {
            // Second child: Sender
            // Use block scope so destructor is called before exit
            {
                InterSimulationInterface<false> sender(shmName);
                sender.exchange(false);
            }  // Destructor called here
            exit(0);
        }

        // Wait for both children to complete
        int receiver_status, sender_status;
        waitpid(receiver_pid, &receiver_status, 0);
        waitpid(sender_pid, &sender_status, 0);

        // Give time for cleanup to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Verify shared memory is cleaned up
        bool shmExists = false;
        try {
            boost::interprocess::managed_shared_memory shmem(boost::interprocess::open_only, shmName.c_str());
            shmExists = true;
        } catch (...) {
            shmExists = false;
        }

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

TEST_F(InterSimulationInterfaceTest, MoveConstructorMaintainsConnection) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with move
        InterSimulationInterface<false> sender1(shmName);
        InterSimulationInterface<false> sender2(std::move(sender1));

        bool received = sender2.exchange(true);
        exit(received ? 1 : 0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(false);
        EXPECT_TRUE(received);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, MoveAssignmentMaintainsConnection) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with move assignment
        InterSimulationInterface<false> sender1(shmName);
        InterSimulationInterface<false> sender2;
        sender2 = std::move(sender1);

        bool received = sender2.exchange(true);
        exit(received ? 1 : 0);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(false);
        EXPECT_TRUE(received);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Edge Cases =====

TEST_F(InterSimulationInterfaceTest, FirstCallBehavior) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - first call should not wait for buffer flip
        InterSimulationInterface<false> sender(shmName);

        auto start = std::chrono::steady_clock::now();
        sender.exchange(true);
        auto end = std::chrono::steady_clock::now();

        // First call should complete quickly (not waiting for previous flip)
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        exit(duration.count() < 100 ? 0 : 1);
    } else {
        // Parent process: Receiver
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        receiver.exchange(false);

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

TEST_F(InterSimulationInterfaceTest, ConsecutiveExchangesSameValue) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Send same value repeatedly
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 50; ++i) {
            sender.exchange(true);  // Always true
        }
        exit(0);
    } else {
        // Parent process: Verify same value received repeatedly
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (int i = 0; i < 50; ++i) {
            bool received = receiver.exchange(false);
            EXPECT_TRUE(received);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Timing and Synchronization Tests =====

TEST_F(InterSimulationInterfaceTest, SynchronizationBetweenProcesses) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender - delayed start
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        InterSimulationInterface<false> sender(shmName);

        for (int i = 0; i < 10; ++i) {
            sender.exchange(true);
        }
        exit(0);
    } else {
        // Parent process: Receiver - starts immediately
        InterSimulationInterface<true> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Should wait for sender to be ready
        for (int i = 0; i < 10; ++i) {
            receiver.exchange(false);
        }

        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}

// ===== Custom Shared Memory Size Tests =====

TEST_F(InterSimulationInterfaceTest, CustomSharedMemorySize) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Sender with larger shared memory
        InterSimulationInterface<false, 4096> sender(shmName);
        sender.exchange(true);
        exit(0);
    } else {
        // Parent process: Receiver with larger shared memory
        InterSimulationInterface<true, 4096> receiver(shmName);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        bool received = receiver.exchange(false);
        EXPECT_TRUE(received);

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
