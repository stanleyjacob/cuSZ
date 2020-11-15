#STRICT_CHECK=-Xcompiler -Wall
#PTX_VERBOSE=-Xptxas -O3,-v

#CXX       := clang++ -fPIE
CXX       := g++
NVCC      := nvcc
STD       := -std=c++14
HOST_DBG  := -O0 -g
CUDA_DBG  := -O0 -G -g
SRC_DIR   := src
OBJ_DIR   := src
BIN_DIR   := bin

GPU_PASCAL:= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61
GPU_VOLTA := -gencode=arch=compute_70,code=sm_70
GPU_TURING:= -gencode=arch=compute_75,code=sm_75
GPU_AMPERE:= -gencode=arch=compute_80,code=sm_80
DEPLOY    := $(GPU_PASCAL) $(GPU_VOLTA)

CUDA_MAJV := $(shell nvcc --version | grep "release" | \
               awk '{print $$6}' | cut -c2- | cut -d. -f1)

ifeq ($(shell test $(CUDA_MAJV) -ge 10; echo $$?), 0)
  DEPLOY += $(GPU_TURING)
endif

ifeq ($(shell test $(CUDA_MAJV) -ge 11; echo $$?), 0)
  DEPLOY += $(GPU_AMPERE)
endif

CCFLAGS   := $(STD) -O3
NVCCFLAGS := $(STD) $(DEPLOY) --expt-relaxed-constexpr
NVCCFLAGS_HUFF := $(STD) $(DEPLOY)

CCFILES_OMP:=$(SRC_DIR)/analysis_utils.cc
CCFILES   := $(filter-out $(CCFILES_OMP), $(wildcard $(SRC_DIR)/*.cc))

MAIN      := $(SRC_DIR)/cusz.cu
CUFILES2  := $(SRC_DIR)/cusz_workflow.cu $(SRC_DIR)/cusz_dualquant.cu
CUFILES3  := $(SRC_DIR)/canonical.cu $(SRC_DIR)/par_merge.cu $(SRC_DIR)/par_huffman.cu
CUFILES1  := $(filter-out $(MAIN) $(CUFILES3) $(CUFILES2), $(wildcard $(SRC_DIR)/*.cu))

CCOBJS_OMP:= $(CCFILES_OMP:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CCOBJS    := $(CCFILES:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CUOBJS1   := $(CUFILES1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS2   := $(CUFILES2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS3   := $(CUFILES3:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CUOBJS    := $(CUOBJS1) $(CUOBJS2) $(CUOBJS3)
OBJS      := $(CCOBJS) $(CCOBJS_OMP) $(CUOBJS)

$(CCOBJS_OMP): CCFLAGS += -fopenmp
# $(CUOBJS1): NVCCFLAGS +=
$(CUOBJS2): NVCCFLAGS += -rdc=true
$(CUOBJS3): NVCCFLAGS += -rdc=true

all: ; @$(MAKE) cusz -j

################################################################################

<<<<<<< HEAD
_DEPS_ARG  := $(SRC_DIR)/argparse.o
_DEPS_MEM  := $(SRC_DIR)/cuda_mem.o
_DEPS_HIST := $(SRC_DIR)/histogram.o $(SRC_DIR)/huffman_workflow.o $(SRC_DIR)/format.o $(SRC_DIR)/canonical.o $(SRC_DIR)/huffman.o -rdc=true
_DEPS_OLDENC := $(SRC_DIR)/huffman_codec.o $(SRC_DIR)/par_huffman.o $(SRC_DIR)/par_huffman_sortbyfreq.o $(SRC_DIR)/par_merge.o
DEPS_HUFF := $(_DEPS_MEM) $(_DEPS_HIST) $(_DEPS_OLDENC) $(_DEPS_ARG)

<<<<<<< HEAD
<<<<<<< HEAD
=======
HUFF_DIR   := $(SRC_DIR)/huffre

_DEPS_ARG  := $(SRC_DIR)/argparse.o
_DEPS_MEM  := $(SRC_DIR)/cuda_mem.o
_DEPS_HIST := $(SRC_DIR)/histogram.o $(SRC_DIR)/huffman_workflow.o $(SRC_DIR)/format.o $(SRC_DIR)/canonical.o $(SRC_DIR)/huffman.o -rdc=true
_DEPS_OLDENC := $(SRC_DIR)/huffman_codec.o
DEPS_HUFF := $(_DEPS_MEM) $(_DEPS_HIST) $(_DEPS_OLDENC) $(_DEPS_ARG)
=======
HUFF_DIR := src/huffre
>>>>>>> fix; change magnitude; rename type
=======
HUFF_DIR := src/huffre
>>>>>>> d0102f5a4a79164e63dbd6a8afd91de75911afb1

huff: $(HUFF_DIR)/huff.cu $(SRC_DIR)/argparse.cc
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huff.cu -o huff
# nvcc huff.cu ../argparse.o ../constants.o ../cuda_mem.o ../huffman_workflow.o ../types.o ../format.o ../histogram.o ../huffman.o ../canonical.o ../huffman_codec.o -gencode=arch=compute_75,code=sm_75

huffre: $(HUFF_DIR)/rs_merge.cuh $(HUFF_DIR)/huffre.cu
	$(NVCC) $(NVCCFLAGS_HUFF) $(DEPS_HUFF) $(HUFF_DIR)/huffre.cu -o huffre

huffretime: $(HUFF_DIR)/rs_merge.cuh $(HUFF_DIR)/huffre.cu
	$(NVCC) $(NVCCFLAGS_HUFF) $(DEPS_HUFF) $(HUFF_DIR)/huffre.cu -o huffre-reduce1  -DREDUCE1TIME
	$(NVCC) $(NVCCFLAGS_HUFF) $(DEPS_HUFF) $(HUFF_DIR)/huffre.cu -o huffre-reduce12 -DREDUCE12TIME
	$(NVCC) $(NVCCFLAGS_HUFF) $(DEPS_HUFF) $(HUFF_DIR)/huffre.cu -o huffre-allmerge -DALLMERGETIME 

huffredbg: $(HUFF_DIR)/rs_merge.cuh $(HUFF_DIR)/huffre.cu
	$(NVCC) $(NVCNVCCFLAGS_HUFFCFLAGS) $(CUDA_DBG) $(DEPS_HUFF) $(HUFF_DIR)/huffre.cu -o huffre-dbg -DDBG0,DBG1,DBG2

<<<<<<< HEAD
>>>>>>> add "Huffman (re)"
=======
>>>>>>> d0102f5a4a79164e63dbd6a8afd91de75911afb1
install: bin/cusz
	cp bin/cusz /usr/local/bin

cusz: $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lgomp -lcusparse $(MAIN) -rdc=true $^ -o $(BIN_DIR)/$@
$(BIN_DIR):
	mkdir $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CXX)  $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)
