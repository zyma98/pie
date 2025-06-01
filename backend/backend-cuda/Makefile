# Makefile for CUDA C++ Project with FlashInfer

# Compiler and flags
NVCC        = nvcc
NVCCFLAGS   = -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra
# Add -g -G for debug symbols and device-side debugging
# NVCCFLAGS   += -g -G

# Directories
SRC_DIR     = src
INCLUDE_DIR = include
LIB_DIR     = lib
BUILD_DIR   = build
BIN_DIR     = bin

# FlashInfer include path
FLASHINFER_INCLUDE_DIR = $(LIB_DIR)/flashinfer/include

# Source files
SOURCES     = $(SRC_DIR)/main.cu
# OBJECTS will be created in the build directory
OBJECTS     = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES))

# Executable name
TARGET      = $(BIN_DIR)/cuda_app

# Include paths for headers
INC_PATH    = -I$(INCLUDE_DIR) -I$(FLASHINFER_INCLUDE_DIR)

# Default target
.PHONY: all
all: $(TARGET)

# Create build and bin directories if they don't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Rule to compile CUDA .cu files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INC_PATH) -c $< -o $@

# Rule to link object files into the final executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Clean up build artifacts
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony target for running the application
.PHONY: run
run: $(TARGET)
	@./$(TARGET)