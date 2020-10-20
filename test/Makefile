
query: test_query.cc ../src/query.hh
	nvcc  -x c++ -I../src/ -DMAIN test_query.cc -o query
all:
	g++ -o test_metadata test_metadata.cc -lgtest -lpthread
