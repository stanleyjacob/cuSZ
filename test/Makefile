
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
<<<<<<< HEAD
=======

argparse:
	g++ -c ../src/format.cc 
	g++ -c ../src/argparse2.cc 
	g++ -c ../src/argparse2_doc.cc 
	g++ -c ../src/argparse2_cusz.cc 
	g++ test_argparse.cc -I../src format.o argparse2.o argparse2_cusz.o argparse2_doc.o -o test_argparse
>>>>>>> change short doc
