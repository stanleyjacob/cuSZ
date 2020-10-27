
query: test_query.cc ../src/query.hh
	nvcc  -x c++ -I../src/ -DMAIN test_query.cc -o query
all:
	g++ -o test_metadata test_metadata.cc -lgtest -lpthread
	nvcc -x c++ -I../src/ -DMAIN test_query.cc -o query

metadata:
	g++ test_metadata.cc -lgtest -lpthread -o metadata 
	
sparse:
	nvcc -std=c++14 -c ../src/impl_mem.cu
	nvcc -std=c++14 -I../src/ test_skewness.cu impl_mem.o -lcusparse -o test_skewness
