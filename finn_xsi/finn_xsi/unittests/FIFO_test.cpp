#include "FIFO.h"

#include <gtest/gtest.h>

// Test fixture for FIFO tests
class FIFOTest : public ::testing::Test {
     protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ===== Constructor and Initialization Tests =====

TEST_F(FIFOTest, ConstructorWithDefaultSize) {
    FIFO fifo;
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getInputReady());
    EXPECT_FALSE(fifo.getOutputValid());
}

TEST_F(FIFOTest, ConstructorWithSpecificSize) {
    FIFO fifo(10);
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getInputReady());
    EXPECT_FALSE(fifo.getOutputValid());
    EXPECT_EQ(fifo.getSpaceLeft(), 10);
}

TEST_F(FIFOTest, ConstructorWithZeroSize) {
    FIFO fifo(0);
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_FALSE(fifo.getInputReady());
    EXPECT_FALSE(fifo.getOutputValid());
    EXPECT_EQ(fifo.getSpaceLeft(), 0);
}

// ===== Reset Tests =====

TEST_F(FIFOTest, ResetClearsState) {
    FIFO fifo(10);
    fifo.update(true, false);  // Add one element
    fifo.toggleClock();
    EXPECT_FALSE(fifo.isEmpty());

    fifo.reset(10);
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_EQ(fifo.getSpaceLeft(), 10);
}

TEST_F(FIFOTest, ResetChangesSize) {
    FIFO fifo(10);
    fifo.reset(20);
    EXPECT_EQ(fifo.getSpaceLeft(), 20);
}

TEST_F(FIFOTest, SetMaxSize) {
    FIFO fifo(10);
    fifo.setMaxSize(15);
    EXPECT_EQ(fifo.getSpaceLeft(), 15);
}

// ===== Basic Update and Toggle Tests =====

TEST_F(FIFOTest, PushOneElement) {
    FIFO fifo(10);
    fifo.update(true, false);  // Push (valid=true, ready=false)
    fifo.toggleClock();

    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getOutputValid());
    EXPECT_TRUE(fifo.getInputReady());
    EXPECT_EQ(fifo.getSpaceLeft(), 9);
}

TEST_F(FIFOTest, PopOneElement) {
    FIFO fifo(10);
    // First push an element
    fifo.update(true, false);
    fifo.toggleClock();

    // Then pop it
    fifo.update(false, true);  // Pop (valid=false, ready=true)
    fifo.toggleClock();

    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_FALSE(fifo.getOutputValid());
    EXPECT_EQ(fifo.getSpaceLeft(), 10);
}

TEST_F(FIFOTest, PushAndPopSimultaneously) {
    FIFO fifo(10);
    // First push an element
    fifo.update(true, false);
    fifo.toggleClock();

    // Now push and pop simultaneously (FIFO size should stay the same)
    fifo.update(true, true);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getOutputValid());
    EXPECT_EQ(fifo.getSpaceLeft(), 9);
}

// ===== Boundary Condition Tests =====

TEST_F(FIFOTest, FillToCapacity) {
    FIFO fifo(3);

    for (int i = 0; i < 3; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
    }

    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getOutputValid());
    EXPECT_FALSE(fifo.getInputReady());
    EXPECT_EQ(fifo.getSpaceLeft(), 0);
}

TEST_F(FIFOTest, CannotPushWhenFull) {
    FIFO fifo(2);

    // Fill the FIFO
    fifo.update(true, false);
    fifo.toggleClock();
    fifo.update(true, false);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.getInputReady());

    // Try to push when full (should have no effect)
    fifo.update(true, false);
    fifo.toggleClock();

    EXPECT_EQ(fifo.getSpaceLeft(), 0);
}

TEST_F(FIFOTest, CanPushAndPullWhenFull) {
    FIFO fifo(2);

    // Fill the FIFO
    fifo.update(true, false);
    fifo.toggleClock();
    fifo.update(true, false);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.getInputReady());

    // Try to push and pull when full (should have no effect)
    fifo.update(true, true);
    fifo.toggleClock();

    EXPECT_EQ(fifo.getSpaceLeft(), 1);

    fifo.reset(2);

    // Fill the FIFO
    fifo.update(true, false);
    fifo.toggleClock();
    fifo.update(true, false);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.getInputReady());

    // Try to push and pull when full (should have no effect)
    fifo.update(false, true);
    fifo.toggleClock();

    EXPECT_EQ(fifo.getSpaceLeft(), 1);
}

TEST_F(FIFOTest, CannotPopWhenEmpty) {
    FIFO fifo(10);

    EXPECT_TRUE(fifo.isEmpty());

    // Try to pop when empty (should have no effect)
    fifo.update(false, true);
    fifo.toggleClock();

    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_EQ(fifo.getSpaceLeft(), 10);
}

TEST_F(FIFOTest, CanPushAndPopWhenEmpty) {
    FIFO fifo(10);

    EXPECT_TRUE(fifo.isEmpty());

    // Try to pop when empty (should have no effect)
    fifo.update(true, true);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getOutputValid());
    EXPECT_EQ(fifo.getSpaceLeft(), 9);
}

TEST_F(FIFOTest, PopWhenFullMakesSpaceAvailable) {
    FIFO fifo(2);

    // Fill the FIFO
    fifo.update(true, false);
    fifo.toggleClock();
    fifo.update(true, false);
    fifo.toggleClock();

    EXPECT_FALSE(fifo.getInputReady());

    // Pop one element
    fifo.update(false, true);
    fifo.toggleClock();

    EXPECT_TRUE(fifo.getInputReady());
    EXPECT_EQ(fifo.getSpaceLeft(), 1);
}

// ===== Sequential Operation Tests =====

TEST_F(FIFOTest, SequentialPushAndPop) {
    FIFO fifo(5);

    // Push 3 elements
    for (int i = 0; i < 3; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.getSpaceLeft(), 2);

    // Pop 2 elements
    for (int i = 0; i < 2; ++i) {
        fifo.update(false, true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.getSpaceLeft(), 4);

    // Pop 1 more
    fifo.update(false, true);
    fifo.toggleClock();
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(FIFOTest, AlternatingPushPop) {
    FIFO fifo(10);

    for (int i = 0; i < 5; ++i) {
        // Push
        fifo.update(true, false);
        fifo.toggleClock();
        EXPECT_FALSE(fifo.isEmpty());

        // Pop
        fifo.update(false, true);
        fifo.toggleClock();
        EXPECT_TRUE(fifo.isEmpty());
    }
}

TEST_F(FIFOTest, StreamingOperation) {
    FIFO fifo(10);

    // Push one element first
    fifo.update(true, false);
    fifo.toggleClock();

    // Now stream: push and pop simultaneously for multiple cycles
    for (int i = 0; i < 100; ++i) {
        fifo.update(true, true);
        fifo.toggleClock();
        EXPECT_EQ(fifo.getSpaceLeft(), 9);  // Size should remain constant
    }
}

// ===== State Query Tests =====

TEST_F(FIFOTest, IsEmptyCorrectly) {
    FIFO fifo(5);
    EXPECT_TRUE(fifo.isEmpty());

    fifo.update(true, false);
    fifo.toggleClock();
    EXPECT_FALSE(fifo.isEmpty());

    fifo.update(false, true);
    fifo.toggleClock();
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(FIFOTest, IsInputReadyCorrectly) {
    FIFO fifo(2);
    EXPECT_TRUE(fifo.getInputReady());

    fifo.update(true, false);
    fifo.toggleClock();
    EXPECT_TRUE(fifo.getInputReady());

    fifo.update(true, false);
    fifo.toggleClock();
    EXPECT_FALSE(fifo.getInputReady());
}

TEST_F(FIFOTest, IsOutputValidCorrectly) {
    FIFO fifo(5);
    EXPECT_FALSE(fifo.getOutputValid());

    fifo.update(true, false);
    fifo.toggleClock();
    EXPECT_TRUE(fifo.getOutputValid());

    fifo.update(false, true);
    fifo.toggleClock();
    EXPECT_FALSE(fifo.getOutputValid());
}

TEST_F(FIFOTest, GetSpaceLeftCorrectly) {
    FIFO fifo(10);
    EXPECT_EQ(fifo.getSpaceLeft(), 10);

    for (int i = 0; i < 3; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
        EXPECT_EQ(fifo.getSpaceLeft(), 10 - i - 1);
    }

    fifo.update(false, true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.getSpaceLeft(), 8);
}

// ===== Edge Case Tests =====

TEST_F(FIFOTest, NoUpdateBeforeToggle) {
    FIFO fifo(10);
    fifo.toggleClock();  // Toggle without update

    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_EQ(fifo.getSpaceLeft(), 10);
}

TEST_F(FIFOTest, LargeCapacity) {
    FIFO fifo(1000000);
    EXPECT_EQ(fifo.getSpaceLeft(), 1000000);

    for (int i = 0; i < 100; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
    }

    EXPECT_EQ(fifo.getSpaceLeft(), 999900);
}

// ===== IncreaseCounter Tests =====

TEST_F(FIFOTest, IncreaseCounterBasic) {
    FIFO fifo(100);

    fifo.update(true, false);
    fifo.toggleClock();
    EXPECT_EQ(fifo.getSpaceLeft(), 99);

    fifo.increaseCounter(5);
    fifo.toggleClock();
    EXPECT_EQ(fifo.getSpaceLeft(), 94);
}

TEST_F(FIFOTest, IncreaseCounterOnEmptyFIFO) {
    FIFO fifo(100);

    fifo.increaseCounter(10);
    fifo.toggleClock();
    EXPECT_EQ(fifo.getSpaceLeft(), 90);
}

TEST_F(FIFOTest, IncreaseCounterZero) {
    FIFO fifo(100);

    fifo.update(true, false);
    fifo.toggleClock();

    fifo.increaseCounter(0);
    fifo.toggleClock();
    EXPECT_EQ(fifo.getSpaceLeft(), 99);
}

// ===== Complex Scenarios =====

TEST_F(FIFOTest, BurstTrafficPattern) {
    FIFO fifo(20);

    // Burst of 10 pushes
    for (int i = 0; i < 10; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.getSpaceLeft(), 10);

    // Burst of 10 pops
    for (int i = 0; i < 10; ++i) {
        fifo.update(false, true);
        fifo.toggleClock();
    }
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(FIFOTest, StressTestManyOperations) {
    FIFO fifo(100);

    // Perform 1000 operations
    for (int i = 0; i < 500; ++i) {
        fifo.update(true, false);
        fifo.toggleClock();
    }

    for (int i = 0; i < 500; ++i) {
        fifo.update(false, true);
        fifo.toggleClock();
    }

    EXPECT_TRUE(fifo.isEmpty());
}

// ===== Multiple FIFO Instances =====

TEST_F(FIFOTest, MultipleFIFOsIndependent) {
    FIFO fifo1(10);
    FIFO fifo2(20);

    fifo1.update(true, false);
    fifo1.toggleClock();

    EXPECT_EQ(fifo1.getSpaceLeft(), 9);
    EXPECT_EQ(fifo2.getSpaceLeft(), 20);

    fifo2.update(true, false);
    fifo2.update(true, false);
    fifo2.toggleClock();
    fifo2.toggleClock();

    // fifo2 should have 2 elements (last update takes effect)
    EXPECT_EQ(fifo1.getSpaceLeft(), 9);
    EXPECT_TRUE(fifo2.getSpaceLeft() < 20);
}

// ===== Individual Method Tests =====

TEST_F(FIFOTest, TryPushBasic) {
    FIFO fifo(10);
    EXPECT_EQ(fifo.size(), 0);

    fifo.setInputValid(true);
    fifo.toggleClock();

    EXPECT_EQ(fifo.size(), 1);
    EXPECT_FALSE(fifo.isEmpty());
    EXPECT_TRUE(fifo.getOutputValid());
}

TEST_F(FIFOTest, TryPushFalseDoesNothing) {
    FIFO fifo(10);

    fifo.setInputValid(false);
    fifo.toggleClock();

    EXPECT_EQ(fifo.size(), 0);
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(FIFOTest, TryPushMultiple) {
    FIFO fifo(10);

    for (int i = 0; i < 5; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }

    EXPECT_EQ(fifo.size(), 5);
    EXPECT_EQ(fifo.getSpaceLeft(), 5);
}

TEST_F(FIFOTest, TryPushWhenFull) {
    FIFO fifo(3);

    // Fill the FIFO
    for (int i = 0; i < 3; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }

    EXPECT_EQ(fifo.size(), 3);
    EXPECT_FALSE(fifo.getInputReady());

    // Try to push when full (should have no effect)
    fifo.setInputValid(true);
    fifo.toggleClock();

    EXPECT_EQ(fifo.size(), 3);
}

TEST_F(FIFOTest, TryPopBasic) {
    FIFO fifo(10);

    // First push an element
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 1);

    // Then pop it
    fifo.setOutputReady(true);
    fifo.toggleClock();

    EXPECT_EQ(fifo.size(), 0);
    EXPECT_TRUE(fifo.isEmpty());
}

TEST_F(FIFOTest, TryPopFalseDoesNothing) {
    FIFO fifo(10);

    fifo.setInputValid(true);
    fifo.toggleClock();

    fifo.setOutputReady(false);
    fifo.toggleClock();

    EXPECT_EQ(fifo.size(), 1);
}

TEST_F(FIFOTest, TryPopWhenEmpty) {
    FIFO fifo(10);

    EXPECT_TRUE(fifo.isEmpty());

    // Try to pop when empty (should have no effect)
    fifo.setOutputReady(true);
    fifo.toggleClock();

    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_EQ(fifo.size(), 0);
}

TEST_F(FIFOTest, TryPushAndTryPopSameCycle) {
    FIFO fifo(10);

    // Push first element
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 1);

    // Push and pop in same cycle (order: push then pop)
    fifo.setInputValid(true);
    fifo.setOutputReady(true);
    fifo.toggleClock();

    // Should still have 1 element (pushed 1, popped 1)
    EXPECT_EQ(fifo.size(), 1);

    // Push and pop in same cycle (order: push then pop)
    fifo.setOutputReady(true);
    fifo.setInputValid(true);
    fifo.toggleClock();

    // Should still have 1 element (pushed 1, popped 1)
    EXPECT_EQ(fifo.size(), 1);
}

TEST_F(FIFOTest, TryPushAndTryPopSameCycleEmptyFIFO) {
    FIFO fifo(10);

    // Push and pop in same cycle (order: push then pop)
    fifo.setInputValid(true);
    fifo.setOutputReady(true);
    fifo.toggleClock();

    // Should still have 1 element (pushed 1, popped 0, because was empty)
    EXPECT_EQ(fifo.size(), 1);

    fifo.reset(10);
    // Push and pop in same cycle (order: push then pop)
    fifo.setInputValid(true);
    fifo.setOutputReady(false);
    fifo.toggleClock();

    // Should still have 0 element (pushed 1, popped 0)
    EXPECT_EQ(fifo.size(), 1);
}

TEST_F(FIFOTest, TryPushAndTryPopSameCycleFullFIFO) {
    FIFO fifo(1);

    // Push first element
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 1);
    EXPECT_FALSE(fifo.getInputReady());

    // Push and pop in same cycle (order: push then pop)
    fifo.setInputValid(true);
    fifo.setOutputReady(true);
    fifo.toggleClock();

    // Should still have 1 element (pushed 1, popped 1)
    EXPECT_EQ(fifo.size(), 0);

    fifo.reset(1);

    // Push first element
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 1);
    EXPECT_FALSE(fifo.getInputReady());

    // Push and pop in same cycle (order: push then pop)
    fifo.setInputValid(false);
    fifo.setOutputReady(true);
    fifo.toggleClock();

    // Should still have 1 element (pushed 1, popped 1)
    EXPECT_EQ(fifo.size(), 0);
}

TEST_F(FIFOTest, TryPushAndTryPopSequence) {
    FIFO fifo(10);

    // Push 3
    for (int i = 0; i < 3; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 3);

    // Pop 2
    for (int i = 0; i < 2; ++i) {
        fifo.setOutputReady(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 1);

    // Push 1 more
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 2);
}

TEST_F(FIFOTest, TryPushAndTryPopStreaming) {
    FIFO fifo(10);

    // Initialize with one element
    fifo.setInputValid(true);
    fifo.toggleClock();

    // Stream: push and pop simultaneously for many cycles
    for (int i = 0; i < 100; ++i) {
        fifo.setInputValid(true);
        fifo.setOutputReady(true);
        fifo.toggleClock();
        EXPECT_EQ(fifo.size(), 1);  // Size should remain constant
    }
}

TEST_F(FIFOTest, TryPushAlternatingValid) {
    FIFO fifo(10);

    for (int i = 0; i < 10; ++i) {
        fifo.setInputValid(i % 2 == 0);  // Push only on even iterations
        fifo.toggleClock();
    }

    EXPECT_EQ(fifo.size(), 5);  // Should have 5 elements
}

TEST_F(FIFOTest, TryPopAlternatingReady) {
    FIFO fifo(10);

    // Fill with 6 elements
    for (int i = 0; i < 6; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }

    // Pop alternating
    for (int i = 0; i < 10; ++i) {
        fifo.setOutputReady(i % 2 == 0);  // Pop only on even iterations
        fifo.toggleClock();
    }

    EXPECT_EQ(fifo.size(), 1);  // 6 - 5 pops = 1
}

TEST_F(FIFOTest, SizeMethodCorrectness) {
    FIFO fifo(20);

    EXPECT_EQ(fifo.size(), 0);

    for (int i = 1; i <= 10; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
        EXPECT_EQ(fifo.size(), i);
    }

    for (int i = 9; i >= 0; --i) {
        fifo.setOutputReady(true);
        fifo.toggleClock();
        EXPECT_EQ(fifo.size(), i);
    }
}

TEST_F(FIFOTest, TryMethodsVsUpdateEquivalence) {
    FIFO fifo1(10);
    FIFO fifo2(10);

    // Use update() on fifo1
    fifo1.update(true, false);  // Push
    fifo1.toggleClock();
    fifo1.update(true, false);  // Push
    fifo1.toggleClock();
    fifo1.update(false, true);  // Pop
    fifo1.toggleClock();

    // Use tryPush/tryPop on fifo2
    fifo2.setInputValid(true);
    fifo2.toggleClock();
    fifo2.setInputValid(true);
    fifo2.toggleClock();
    fifo2.setOutputReady(true);
    fifo2.toggleClock();

    // Should have same result
    EXPECT_EQ(fifo1.size(), fifo2.size());
    EXPECT_EQ(fifo1.isEmpty(), fifo2.isEmpty());
    EXPECT_EQ(fifo1.getOutputValid(), fifo2.getOutputValid());
}

TEST_F(FIFOTest, TryMethodsBurstPattern) {
    FIFO fifo(50);

    // Burst of pushes
    for (int i = 0; i < 30; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 30);

    // Burst of pops
    for (int i = 0; i < 20; ++i) {
        fifo.setOutputReady(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 10);

    // Mixed burst
    for (int i = 0; i < 15; ++i) {
        fifo.setInputValid(true);
        fifo.setOutputReady(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 10);  // Should remain constant
}

TEST_F(FIFOTest, TryMethodsStressTest) {
    FIFO fifo(1000);

    // Complex pattern
    for (int i = 0; i < 500; ++i) {
        fifo.setInputValid(i % 3 != 0);  // Push 2 out of 3 times
        if (i > 100) {
            fifo.setOutputReady(i % 2 == 0);  // Pop every other time after 100
        }
        fifo.toggleClock();
    }

    // Verify FIFO is in valid state
    EXPECT_LE(fifo.size(), 1000);
    EXPECT_EQ(fifo.size() == 0, fifo.isEmpty());
    EXPECT_EQ(fifo.size() > 0, fifo.getOutputValid());
}

TEST_F(FIFOTest, TryMethodsEdgeCaseFullToEmpty) {
    FIFO fifo(5);

    // Fill completely
    for (int i = 0; i < 5; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 5);
    EXPECT_FALSE(fifo.getInputReady());

    // Empty completely
    for (int i = 0; i < 5; ++i) {
        fifo.setOutputReady(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 0);
    EXPECT_TRUE(fifo.isEmpty());
    EXPECT_FALSE(fifo.getOutputValid());
}

TEST_F(FIFOTest, TryMethodsWithReset) {
    FIFO fifo(10);

    // Add some elements
    for (int i = 0; i < 5; ++i) {
        fifo.setInputValid(true);
        fifo.toggleClock();
    }
    EXPECT_EQ(fifo.size(), 5);

    // Reset
    fifo.reset(10);
    EXPECT_EQ(fifo.size(), 0);

    // Should work normally after reset
    fifo.setInputValid(true);
    fifo.toggleClock();
    EXPECT_EQ(fifo.size(), 1);
}

TEST_F(FIFOTest, TestTimeout){
    FIFO fifo(10);
    fifo.setCyclesUntilExpectedFirstValid(3);
    EXPECT_TRUE(fifo.toggleClock());  // 3 cycles left
    EXPECT_TRUE(fifo.toggleClock());  // 2 cycles left
    EXPECT_FALSE(fifo.toggleClock());  // 0 cycles left, should return false

    fifo.reset(10);
    fifo.setCyclesUntilExpectedFirstValid(2);
    EXPECT_TRUE(fifo.toggleClock());  // 2 cycles left
    fifo.update(true, false);        // Set valid, should disable timeout
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true

    fifo.reset(10);
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
    EXPECT_TRUE(fifo.toggleClock());  // Still should return true
}

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
