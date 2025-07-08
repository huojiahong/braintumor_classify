cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst src/%.cpp,obj/%.o,$(cpp_srcs))

include_dirs := /home/hjh/modules/onnxruntime-linux-x64-1.20.1/include /usr/local/include/opencv4/
library_dirs := /home/hjh/modules/onnxruntime-linux-x64-1.20.1/lib /usr/local/lib
linking_libs := onnxruntime opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs

I_options := $(include_dirs:%=-I%)
L_options := $(library_dirs:%=-L%)
l_options := $(linking_libs:%=-l%)
r_options := $(library_dirs:%=-Wl,-rpath=%)

compile_options := -g -w -O3 $(I_options)
linking_options := $(L_options) $(l_options) $(r_options)

obj/%.o : src/%.cpp
	@echo [INFO]: Compile $^ to $@
	@mkdir -p $(dir $@)
	@g++ -c $^ -o $@ $(compile_options)

workspace/exec : $(cpp_objs)
	@echo [INFO]: Link $^ to $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(linking_options)

run : workspace/exec
	@./$<

compile : $(cpp_objs)

debug:
	@echo $(cpp_srcs)
	@echo $(cpp_objs)
	@echo $(I_options)

clean:
	@rm -rf obj workspace/exec

.PHONY : debug run clean