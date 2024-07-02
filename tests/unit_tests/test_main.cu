#include "test_chol.cuh"
#include "test_copy.cuh"
#include "test_matrix.cuh"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}