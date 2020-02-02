LIBTORCH := "`pwd`/libtorch"
LD_LIBRARY_PATH := "$(LIBTORCH)/lib:$(LD_LIBRARY_PATH)"
C_INCLUDE_PATH := "$(LIBTORCH)/include:$(C_INCLUDE_PATH)"

$(LIBTORCH)/lib/libtorch:
	@echo "Download libtorch ."
	curl -O "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.3.1.zip"
	unzip libtorch-macos-1.3.1.zip

build:
	cargo build

echo:
	@echo "$(C_INCLUDE_PATH)   $(LD_LIBRARY_PATH)"