rm dw_test
make clean && make -j8 dw_test
./dw_test --gtest_filter=Partition*
