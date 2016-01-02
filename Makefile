# Makefile for sampler

.DEFAULT_GOAL := $(PROGRAM)

# common compiler flags
CXXFLAGS = -std=c++0x -Wall -fno-strict-aliasing
LDFLAGS = -L./lib/zeromq/lib
LDLIBS = -lzmq
CXXFLAGS += -I./lib/tclap/include/ -I./src -I./lib/zeromq/include/

# platform dependent compiler flags
UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
ifndef CXX
CXX = g++
endif
LDFLAGS += -Wl,-rpath,`pwd`/lib/zeromq/lib
# optimization flags
CXXFLAGS += -Ofast
# using NUMA for Linux
CXXFLAGS += -I./lib/numactl-2.0.9/
LDFLAGS += -L./lib/numactl-2.0.9/
LDFLAGS += -Wl,-Bstatic -Wl,-Bdynamic
LDLIBS += -lnuma -lrt -lpthread
endif

ifeq ($(UNAME), Darwin)
ifndef CXX
CXX = clang++
endif
# optimization
CXXFLAGS += -O3 -stdlib=libc++ -mmacosx-version-min=10.7
CXXFLAGS += -flto
endif

# source files
SOURCES += src/gibbs.cpp
SOURCES += src/io/cmd_parser.cpp
SOURCES += src/io/binary_parser.cpp
SOURCES += src/main.cpp
SOURCES += src/dstruct/factor_graph/weight.cpp
SOURCES += src/dstruct/factor_graph/variable.cpp
SOURCES += src/dstruct/factor_graph/factor.cpp
SOURCES += src/dstruct/factor_graph/factor_graph.cpp
SOURCES += src/dstruct/factor_graph/inference_result.cpp
SOURCES += src/app/gibbs/gibbs_sampling.cpp
SOURCES += src/app/gibbs/single_thread_sampler.cpp
SOURCES += src/app/gibbs/single_node_sampler.cpp
SOURCES += src/timer.cpp
OBJECTS = $(SOURCES:.cpp=.o)
PROGRAM = dw

# test files
TEST_SOURCES += test/test.cpp
TEST_SOURCES += test/FactorTest.cpp
TEST_SOURCES += test/LogisticRegressionTest.cpp
TEST_SOURCES += test/binary_parser_test.cpp
TEST_SOURCES += test/loading_test.cpp
TEST_SOURCES += test/factor_graph_test.cpp
TEST_SOURCES += test/sampler_test.cpp
TEST_SOURCES += test/multinomial.cpp
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
TEST_PROGRAM = $(PROGRAM)_test
# test files need gtest
$(TEST_OBJECTS): CXXFLAGS += -I./lib/gtest-1.7.0/include/
$(TEST_PROGRAM): LDFLAGS += -L./lib/gtest/
$(TEST_PROGRAM): LDLIBS += -lgtest -lpthread

# how to link our sampler
$(PROGRAM): $(OBJECTS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LDLIBS)

# how to link our sampler unit tests
$(TEST_PROGRAM): $(TEST_OBJECTS) $(filter-out src/main.o,$(OBJECTS))
	$(CXX) -o $@ $(LDFLAGS) $^ $(LDLIBS)

# how to compile each source
%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(CXXFLAGS) -c $<

# how to get dependencies prepared
dep:
	# gtest for tests
	cd lib;\
	unzip gtest-1.7.0.zip;\
	mkdir gtest;\
	cd gtest;\
	cmake ../gtest-1.7.0;\
	make
	# tclap for command-line args parsing
	cd lib;\
	tar xf tclap-1.2.1.tar.gz;\
	cd tclap-1.2.1;\
	./configure --prefix=`pwd`/../tclap;\
	make;\
	make install
	# libsodium
	cd lib;\
	wget https://download.libsodium.org/libsodium/releases/libsodium-1.0.2.tar.gz;\
	tar xf libsodium-1.0.2.tar.gz;\
	cd libsodium-1.0.2;\
	./configure --prefix=`pwd`/../libsodium;\
	make;\
	make install
	# zmq
	cd lib;\
	wget http://download.zeromq.org/zeromq-4.1.3.tar.gz;\
	tar xf zeromq-4.1.3.tar.gz;\
	cd zeromq-4.1.3;\
	PKG_CONFIG_PATH=`pwd`/../libsodium/lib/pkgconfig ./configure --prefix=`pwd`/../zeromq;\
	make;\
	make install;\
	wget https://raw.githubusercontent.com/zeromq/cppzmq/4d79066be3c3ed0cbf247d3a2eb04be2b803eb3c/zmq.hpp;\
	mv zmq.hpp ../zeromq/include/
	# bats for end-to-end tests
	git clone https://github.com/sstephenson/bats test/bats
.PHONY: dep

# how to clean
clean:
	rm -f $(PROGRAM) $(OBJECTS) $(TEST_PROGRAM) $(TEST_OBJECTS)
.PHONY: clean

# how to test
test: unit-test end2end-test
unit-test: $(TEST_PROGRAM)
	./$(TEST_PROGRAM)
PATH := $(shell pwd)/test/bats/bin:$(PATH)
end2end-test: $(PROGRAM)
	bats test/*.bats
.PHONY: test unit-test end2end-test
