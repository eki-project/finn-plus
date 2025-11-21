#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>

#include "FIFO.h"
#include "InterSimulationInterface.hpp"

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
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = false;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==false for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyTrueValidFalse) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = false;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyFalseValidTrue) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {  // It is correct that valid is false here, because we only have a single cycle and the fifo input is set to valid in cycle 0. Therefore, the FIFO
                                // output is valid in cycle 1 and we should receive a valid in cycle 1.
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = true;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, OneCycleReadyTrueValidTrue) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {  // It is correct that valid is false here, because we only have a single cycle and the fifo input is set to valid in cycle 0. Therefore, the FIFO
                                // output is valid in cycle 1 and we should receive a valid in cycle 1.
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }
        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = true;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

// ===== Multicycle Integration Tests =====

TEST_F(IntegrationTest, TwoCycleReadyFalseValidFalse) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(false);
            readySignal = simDummy.isInputReady();  // Should be false now
            validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = false;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        EXPECT_TRUE(inputFifo.isInputReady());
        incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_FALSE(incomingReady);  // We are in cycle 1; expect ready==false for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, TwoCycleReadyTrueValidFalse) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(true);
            readySignal = simDummy.isInputReady();  // Should be true now
            validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = false;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        EXPECT_TRUE(inputFifo.isInputReady());
        incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(IntegrationTest, TwoCycleReadyFalseValidTrue) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(false);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(false);
            readySignal = simDummy.isInputReady();  // Should be false now
            validSignal = receiver.exchange(readySignal);
            if (!validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify we received all data
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = true;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());
        EXPECT_TRUE(inputFifo.isOutputValid());
        incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_FALSE(incomingReady);  // We are in cycle 1; expect ready==false for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 13);
        EXPECT_TRUE(inputFifo.isInputReady());

    }  // Destructor called here

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_EQ(WEXITSTATUS(status), 0);
}


TEST_F(IntegrationTest, TwoCycleReadyTrueValidTrue) {
    // Test: FIFO feeds data to InterSimulationInterface sender/receiver pair
    // Architecture: FIFO (process A) -> Sender -> Receiver (process B) -> SimDummy -> validation

    pid_t pid = fork();

    if (pid == 0) {
        // Child process: Receiver with FIFO output validation
        int receivedCount = 0;
        {
            InterSimulationInterface<true> receiver(shmName);
            SimDummy simDummy;

            simDummy.setNextReady(true);
            bool readySignal = simDummy.isInputReady();
            bool validSignal = receiver.exchange(readySignal);
            if (validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 1 STARTS

            // Verify we received all data
            if (simDummy.isOutputValid()) {
                exit(1);
            }

            simDummy.setNextReady(true);
            readySignal = simDummy.isInputReady();  // Should be true now
            validSignal = receiver.exchange(readySignal);
            if (!validSignal) {
                exit(2);
            }
            simDummy.setNextValid(validSignal);
            simDummy.toggleClock();  // BELOW HERE CYCLE 2 STARTS

            // Verify we received all data
            if (!simDummy.isOutputValid()) {
                exit(1);
            }

        }  // Destructor called here
        exit(0);
    }

    // Parent process: Sender with FIFO input
    {
        InterSimulationInterface<false> sender(shmName);
        FIFO inputFifo(15);

        bool validSignal = true;
        EXPECT_TRUE(inputFifo.isInputReady());
        bool incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 0; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 15);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 1 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());
        EXPECT_TRUE(inputFifo.isOutputValid());
        incomingReady = sender.exchange(inputFifo.isOutputValid());
        EXPECT_TRUE(incomingReady);  // We are in cycle 1; expect ready==true for cycle 1
        inputFifo.update(validSignal, incomingReady);
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());
        inputFifo.toggleClock();  // BELOW HERE CYCLE 2 STARTS
        EXPECT_EQ(inputFifo.getSpaceLeft(), 14);
        EXPECT_TRUE(inputFifo.isInputReady());

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

    //Propagate valid through SimDummy
    sim.setNextValid(true);
    fifo.update(sim.isOutputValid(), false);
    EXPECT_TRUE(fifo.isInputReady());
    sim.setNextReady(fifo.isInputReady());
    fifo.toggleClock();
    sim.toggleClock();
    EXPECT_EQ(fifo.size(), 0);
    EXPECT_TRUE(sim.isInputReady());

    //Fill FIFO to capacity
    for (std::size_t i = 0; i < 15; ++i) {
        sim.setNextValid(true);

        fifo.update(sim.isOutputValid(), false);
        EXPECT_TRUE(fifo.isInputReady());
        sim.setNextReady(fifo.isInputReady());
        EXPECT_EQ(fifo.size(), i);
        fifo.toggleClock();
        sim.toggleClock();
        EXPECT_EQ(fifo.size(), i+1);
        EXPECT_TRUE(sim.isInputReady());
    }

    EXPECT_FALSE(fifo.isInputReady()); // FIFO changed to not ready on this cycle; Sim is still ready
    sim.setNextValid(true);
    fifo.update(sim.isOutputValid(), false);
    EXPECT_FALSE(fifo.isInputReady());
    sim.setNextReady(fifo.isInputReady());
    fifo.toggleClock();
    sim.toggleClock(); //Propagate ready false through sim

    EXPECT_EQ(fifo.size(), 15);
    EXPECT_FALSE(sim.isInputReady());
}
