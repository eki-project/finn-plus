#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>

#include "FIFO.h"
#include "InterprocessCommunicationChannel.hpp"

// Test fixture for integration tests
class IntegrationTest : public ::testing::Test {
     protected:
    std::string shmName;

    void SetUp() override {
        // Generate unique shared memory name for each test
        shmName = "test_shm_integration_" + std::to_string(getpid());

        // Clean up any leftover shared memory from previous runs
        boost::interprocess::shared_memory_object::remove(shmName.c_str());
    }

    void TearDown() override {
        // Clean up shared memory after test
        boost::interprocess::shared_memory_object::remove(shmName.c_str());
    }
};

class SimDummy {
    bool currentValid = false;
    bool currentReady = true;
    bool nextValid = false;
    bool nextReady = true;

     public:
    bool isOutputValid() const { return currentValid; }
    void toggleClock() {
        currentValid = nextValid;
        currentReady = nextReady;
    }
    bool isInputReady() const { return currentReady; }
    void setNextValid(bool v) { nextValid = v; }
    void setNextReady(bool r) { nextReady = r; }
};

// ===== Basic Integration Tests =====

TEST_F(IntegrationTest, OneCycleReadyFalseValidFalse) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            if (outputFifo.getInputReady() != true) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = false;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==false for cycle 1

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyTrueValidFalse) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            if (outputFifo.getInputReady() != true) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = false;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyFalseValidTrue) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (!validSignal) {  // It is correct that valid is true here, because we only have a single cycle and the sender input is set to valid in cycle 0. Therefore, we should
                                 // receive a valid in cycle 0.
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 14) {
                exit(3);
            }
            if (outputFifo.getInputReady() != true) {
                exit(4);
            }
            if (!outputFifo.getOutputValid()) {
                exit(5);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = true;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyTrueValidTrue) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (!validSignal) {  // It is correct that valid is true here, because we only have a single cycle and the sender input is set to valid in cycle 0. Therefore, we should
                                 // receive a valid in cycle 0.
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 14) {
                exit(3);
            }
            if (outputFifo.getInputReady() != true) {
                exit(4);
            }
            if (!outputFifo.getOutputValid()) {
                exit(5);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = true;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// ===== Multicycle Integration Tests =====

TEST_F(IntegrationTest, TwoCycleReadyFalseValidFalse) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(false);
            readySignal = outputFifo.getInputReady();  // Should be true
            validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = false;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 2

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, TwoCycleReadyTrueValidFalse) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(true);
            readySignal = outputFifo.getInputReady();  // Should be true now
            validSignal = receiver.receive_request();
            if (validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 15) {
                exit(3);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = false;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 2

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, TwoCycleReadyFalseValidTrue) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (!validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 14) {
                exit(3);
            }
            if (!outputFifo.getOutputValid()) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(false);
            readySignal = outputFifo.getInputReady();  // Should be true now (FIFO not full)
            validSignal = receiver.receive_request();
            if (!validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 13) {
                exit(3);
            }
            if (!outputFifo.getOutputValid()) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = true;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 2

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}


TEST_F(IntegrationTest, TwoCycleReadyTrueValidTrue) {
    // Test: FIFO feeds data to InterprocessCommunicationChannel sender/receiver pair
    // Architecture: Sender (process A) -> Receiver -> FIFO (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output to SimDummy
        int receivedCount = 0;
        {
            InterprocessCommunicationChannel<bool, bool, false> receiver(shmName);
            FIFO outputFifo(15);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = outputFifo.getInputReady();
            bool validSignal = receiver.receive_request();
            if (!validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify FIFO state and SimDummy
            if (outputFifo.getSpaceLeft() != 14) {
                exit(3);
            }
            if (!outputFifo.getOutputValid()) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(true);
            readySignal = outputFifo.getInputReady();  // Should be true now
            validSignal = receiver.receive_request();
            if (!validSignal) {
                exit(2);
            }
            receiver.send_response(readySignal);
            outputFifo.update(validSignal, simDummy.isInputReady());
            outputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify FIFO state and SimDummy - FIFO consumes data because SimDummy is ready
            if (outputFifo.getSpaceLeft() != 14) {
                exit(3);
            }
            if (!outputFifo.getOutputValid()) {
                exit(4);
            }
            simDummy.setNextValid(outputFifo.getOutputValid());
            simDummy.toggleClock();
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender
    {
        InterprocessCommunicationChannel<bool, bool, true> sender(shmName);

        bool validSignal = true;
        bool incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        incomingReady = sender.send_request(validSignal);
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 2

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}


// ===== Sender Side Integration Tests =====

TEST_F(IntegrationTest, SimToFIFO) {
    // Architecture: SimDummy -> FIFO

    SimDummy sim;
    FIFO fifo(15);

    // Propagate valid through SimDummy
    sim.setNextValid(true);
    fifo.update(sim.isOutputValid(), false);
    EXPECT_TRUE(fifo.getInputReady());
    sim.setNextReady(fifo.getInputReady());
    fifo.toggleClock();
    sim.toggleClock();
    EXPECT_EQ(fifo.size(), 0);
    EXPECT_TRUE(sim.isInputReady());

    // Fill FIFO to capacity
    for (std::size_t i = 0; i < 15; ++i) {
        sim.setNextValid(true);

        fifo.update(sim.isOutputValid(), false);
        EXPECT_TRUE(fifo.getInputReady());
        sim.setNextReady(fifo.getInputReady());
        EXPECT_EQ(fifo.size(), i);
        fifo.toggleClock();
        sim.toggleClock();
        EXPECT_EQ(fifo.size(), i + 1);
        EXPECT_TRUE(sim.isInputReady());
    }

    EXPECT_FALSE(fifo.getInputReady());  // FIFO changed to not ready on this cycle; Sim is still ready
    sim.setNextValid(true);
    fifo.update(sim.isOutputValid(), false);
    EXPECT_FALSE(fifo.getInputReady());
    sim.setNextReady(fifo.getInputReady());
    fifo.toggleClock();
    sim.toggleClock();  // Propagate ready false through sim

    EXPECT_EQ(fifo.size(), 15);
    EXPECT_FALSE(sim.isInputReady());
}
