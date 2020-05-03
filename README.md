# People Counter (AI on the Edge)<!-- omit in toc -->

### This Project is a submission of the first project in **Intel Edge AI Nanodegree** at Udacity.<!-- omit in toc -->

# Index<!-- omit in toc -->

- [Introduction](#introduction)
  - [What it Does](#what-it-does)
  - [How it Works](#how-it-works)
  - [Requirements (Only for this code)](#requirements-only-for-this-code)
    - [Hardware](#hardware)
    - [Software](#software)
- [How to run](#how-to-run)
    - [Follow the Instruction to Setup your environment given here](#follow-the-instruction-to-setup-your-environment-given-here)
    - [Step 1 - Start the Mosca server](#step-1---start-the-mosca-server)
    - [Step 2 - Start the GUI](#step-2---start-the-gui)
    - [Step 3 - FFmpeg Server](#step-3---ffmpeg-server)
    - [Step 4 - Run the code](#step-4---run-the-code)
      - [Setup the environment](#setup-the-environment)
      - [Running on CPU (Default)](#running-on-cpu-default)
      - [Viewing the App in your Browser](#viewing-the-app-in-your-browser)
- [Project Write-Up](#project-write-up)
  - [What are Custom Layers](#what-are-custom-layers)
    - [To convert a Custom Layer (Tensorflow example)](#to-convert-a-custom-layer-tensorflow-example)
      - [Step 1 - Create the Custom Layer](#step-1---create-the-custom-layer)
        - [Generate the Extension Template Using the Model Extension Generator](#generate-the-extension-template-using-the-model-extension-generator)
      - [Step 2 - Using Model Optimizer to Generate IR Files Containing the Custom Layer](#step-2---using-model-optimizer-to-generate-ir-files-containing-the-custom-layer)
        - [Edit the Extractor Extension Template File](#edit-the-extractor-extension-template-file)
      - [Step 3 - Edit the Operation Extension Template File](#step-3---edit-the-operation-extension-template-file)
      - [Step 4 - Generate the Model IR Files](#step-4---generate-the-model-ir-files)
    - [Inference Engine Custom Layer Implementation for the Intel® CPU](#inference-engine-custom-layer-implementation-for-the-intel%c2%ae-cpu)
      - [Edit the CPU Extension Template Files](#edit-the-cpu-extension-template-files)
        - [Edit `ext_CustomLayer.cpp`](#edit-extcustomlayercpp)
        - [Edit `CMakeLists.txt`](#edit-cmakeliststxt)
      - [Compile the Extension Library](#compile-the-extension-library)
    - [Execute the Model with the Custom Layer](#execute-the-model-with-the-custom-layer)
      - [Using a C++ Sample](#using-a-c-sample)
      - [Using a Python Sample](#using-a-python-sample)
    - [Why you might need to handle custom layers?](#why-you-might-need-to-handle-custom-layers)
  - [Comparing Model Performance](#comparing-model-performance)
  - [Assess Model Use Cases](#assess-model-use-cases)
  - [Assess Effects on End-User Needs](#assess-effects-on-end-user-needs)
  - [Model Research](#model-research)

# Introduction

## What it Does

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in the frame, and total count.

## How it Works

The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used should be able to identify people in a video frame. The app should count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.

You will choose a model to use and convert it with the Model Optimizer.

![architectural diagram](./images/arch_diagram.png)

## Requirements (Only for this code)

### Hardware

* 6th to 10th generation Intel® Core™ processor

### Software

*   Intel® Distribution of OpenVINO™ toolkit 2020 R1 release
*   Node v6.17.1
*   Npm v3.10.10
*   CMake
*   MQTT Mosca server

# How to run

### Follow the Instruction to Setup your environment given [here](Project_Default_Instructions.md/#Setup)
Once you have successfully set-up your environment. From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

*You should see the following message, if successful:*
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

*You should see the following message in the terminal.*
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

#### Running on CPU (Default)

Run the following command

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph_frcnn.xml -d CPU -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

*(Note: This command is only for running inference on CPU in Linux environment with Faster R-CNN Inception V2 on the given video file in resources, for any changes refer to default project instructions.)*

#### Viewing the App in your Browser

To see the output on a web based interface, open the link http://0.0.0.0:3004 in a browser.

*(Note: If the above URL Redirects to Intel's Github, try visiting http://0.0.0.0:3000 or redirecting the stream to Port 3005.)*

<br><br><br><br>

# Project Write-Up

## What are Custom Layers

OpenVINO Toolkit Documentations has a [list of Supported Framework Layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) for DL Inference. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

*(Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices. [Learn more about it here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).)*

### To convert a Custom Layer (Tensorflow example)

#### Step 1 - Create the Custom Layer

##### Generate the Extension Template Using the Model Extension Generator

Model Extension Generator tool will automatically create templates for all the extensions needed by the Model Optimizer to convert and the Inference Engine to execute the custom layer. The extension template files will be partially replaced by Python and C++ code to implement the functionality of your custom layer as needed by the different tools. To create the four extensions for the custom layer, we run the Model Extension Generator with the following options:

- `--mo-tf-ext` = Generate a template for a Model Optimizer Tensorflow extractor
- `--mo-op` = Generate a template for a Model Optimizer custom layer operation
- `--ie-cpu-ext` = Generate a template for an Inference Engine CPU extension
- `--ie-gpu-ext` = Generate a template for an Inference Engine GPU extension
- `--output-dir` = Set the output directory. Here we are using your/directory/cl_CustomLayer as the target directory to store the output from the Model Extension Generator.

To Create the four extension templates for the custom layer, run the command 

```bash
python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir = your/directory
```

The Model Extension Generator will start in interactive mode and prompt us with questions 
about the custom layer to be generated.  Use the text between the `[]`'s to answer each 
of the Model Extension Generator questions as follows:

```
Enter layer name: 
[layer_name]

Do you want to automatically parse all parameters from the model file? (y/n)
...
[n]

Enter all parameters in the following format:
...
Enter 'q' when finished:
[q]

Do you want to change any answer (y/n) ? Default 'no'
[n]

Do you want to use the layer name as the operation name? (y/n)
[y]

Does your operation change shape? (y/n)  
[n]

Do you want to change any answer (y/n) ? Default 'no'
[n]
```

When complete, the output text will appear similar to:
```
Stub file for TensorFlow Model Optimizer extractor is in your/directory/user_mo_extensions/front/tf folder
Stub file for the Model Optimizer operation is in your/directory/user_mo_extensions/ops folder
Stub files for the Inference Engine CPU extension are in your/directory/user_ie_extensions/cpu folder
Stub files for the Inference Engine GPU extension are in your/directory/user_ie_extensions/gpu folder
```
Template files (containing source code stubs) that may need to be edited have just been 
created in the following locations:

- TensorFlow Model Optimizer extractor extension: 
  - `your/directory/user_mo_extensions/front/tf/`
  - `CustomLayer_ext.py`
- Model Optimizer operation extension:
  - `your/directory/user_mo_extensions/ops`
  - `CustomLayer.py`
- Inference Engine CPU extension:
  - `your/directory/user_ie_extensions/cpu`
  - `ext_CustomLayer.cpp`
  - `CMakeLists.txt`
- Inference Engine GPU extension:
  - `your/directory/user_ie_extensions/gpu`
  - `CustomLayer_kernel.cl`
  - `CustomLayer_kernel.xml`


#### Step 2 - Using Model Optimizer to Generate IR Files Containing the Custom Layer 

Now use the generated extractor and operation extensions with the Model Optimizer 
to generate the model IR files needed by the Inference Engine.  The steps covered are:

1. Edit the extractor extension template file
2. Edit the operation extension template file
3. Generate the Model IR Files

##### Edit the Extractor Extension Template File

Below is a walkthrough of  the Python code for the extractor extension that appears in the file 
`your/directory/user_mo_extensions/front/tf/CustomLayer_ext.py`.
1. Using the text editor, open the extractor extension source file `your/directory/user_mo_extensions/front/tf/CustomLayer_ext.py`.
2. The class is defined with the unique name `CustomLayerFrontExtractor` that inherits from the base extractor `FrontExtractorOp` class.  The class variable `op` is set to the name of the layer operation and `enabled` is set to tell the Model Optimizer to use (`True`) or exclude (`False`) the layer during processing.

    ```python
    class CustomLayerFrontExtractor(FrontExtractorOp):
        op = 'CustomLayer' 
        enabled = True
    ```

3. The `extract` function is overridden to allow modifications while extracting parameters from layers within the input model.

    ```python
    @staticmethod
    def extract(node):
    ```

4. The layer parameters are extracted from the input model and stored in `param`.  This is where the layer parameters in `param` may be retrieved and used as needed.  For a simple custom layer, the `op` attribute is simply set to the name of the operation extension used.

    ```python
    proto_layer = node.pb
    param = proto_layer.attr
    # extracting parameters from TensorFlow layer and prepare them for IR
    attrs = {
        'op': __class__.op
    }
    ```

5. The attributes for the specific node are updated. This is where we can modify or create attributes in `attrs` before updating `node` with the results and the `enabled` class variable is returned.

    ```python
    # update the attributes of the node
    Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
    
    return __class__.enabled
    ```

#### Step 3 - Edit the Operation Extension Template File

If the shape (i.e., dimensions) of the layer output is the same as the input shape, the generated operation extension does not need to be modified.  
Below is a walkthrough of the Python code for the operation extension that appears in 
the file  `your/directory/user_mo_extensions/ops/CustomLayer.py`.

1. Using the text editor, open the operation extension source file `your/directory/user_mo_extensions/ops/CustomLayer.py` 
2. The class is defined with the unique name `CustomLayerOp` that inherits from the base operation `Op` class.  The class variable `op` is set to `'CustomLayer'`, the name of the layer operation.

    ```python
    class CustomLayerOp(Op):
    op = 'CustomLayer'
    ```

3. The `CustomLayerOp` class initializer `__init__` function will be called for each layer created.  The initializer must initialize the super class `Op` by passing the `graph` and `attrs` arguments along with a dictionary of the mandatory properties for the `CustomLayer` operation layer that define the type (`type`), operation (`op`), and inference function (`infer`).  This is where any other initialization needed by the `CustomLayerOP` operation can be specified.

    ```python
    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            infer=CustomLayerOp.infer            
        )
    super().__init__(graph, mandatory_props, attrs)
    ```

4. The `infer` function is defined to provide the Model Optimizer information on a layer, specifically returning the shape of the layer output for each node.  Here, the layer output shape is the same as the input and the value of the helper function `copy_shape_infer(node)` is returned.

    ```python
    @staticmethod
    def infer(node: Node):
        # ==========================================================
        # You should add your shape calculation implementation here
        # If a layer input shape is different to the output one
        # it means that it changes shape and you need to implement
        # it on your own. Otherwise, use copy_shape_infer(node).
        # ==========================================================
        return copy_shape_infer(node)
    ```

#### Step 4 - Generate the Model IR Files

With the extensions now complete, we use the Model Optimizer to convert and optimize 
the example TensorFlow model into IR files that will run inference using the Inference Engine.  
To create the IR files, we run the Model Optimizer for TensorFlow `mo_tf.py` with 
the following options:

- `--input_meta_graph model.ckpt.meta`
  - Specifies the model input file.  

- `--batch 1`
  - Explicitly sets the batch size to 1 because the example model has an input dimension of "-1".
  - TensorFlow allows "-1" as a variable indicating "to be filled in later", however the Model Optimizer requires explicit information for the optimization process.  

- `--output "ModCustomLayer/Activation_8/softmax_output"`
  - The full name of the final output layer of the model.

- `--extensions your/director/user_mo_extensions`
  - Location of the extractor and operation extensions for the custom layer to be used by the Model Optimizer during model extraction and optimization. 

- `--output_dir your/directory/cl_ext_CustomLayer`
  - Location to write the output IR files.

To create the model IR files that will include the `CustomLayer` custom layer, we run the commands:

```bash
cd your/directory/tf_model
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCustomLayer/Activation_8/softmax_output" --extensions your/directory/cl_CustomLayer/user_mo_extensions --output_dir your/directory/cl_ext_CustomLayer
```

The output will appear similar to:

```
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: your/directory/cl_ext_CustomLayer/model.ckpt.xml
[ SUCCESS ] BIN file: your/directory/cl_ext_CustomLayer/model.ckpt.bin
[ SUCCESS ] Total execution time: x.xx seconds.
```

### Inference Engine Custom Layer Implementation for the Intel® CPU

We will now use the generated CPU extension with the Inference Engine to execute 
the custom layer on the CPU.  The steps are:

1. Edit the CPU extension template files.
2. Compile the CPU extension library.
3. Execute the Model with the custom layer.

You *will* need to make the changes in this section to the related files.

Note that the classroom workspace only has an Intel CPU available, so we will not perform
the necessary steps for GPU usage with the Inference Engine.

#### Edit the CPU Extension Template Files

The generated CPU extension includes the template file `ext_CustomLayer.cpp` that must be edited 
to fill-in the functionality of the `CustomLayer` custom layer for execution by the Inference Engine.  
We also need to edit the `CMakeLists.txt` file to add any header file or library dependencies 
required to compile the CPU extension.  In the next sections, we will walk through and edit 
these files.

##### Edit `ext_CustomLayer.cpp`

Now edit the `ext_CustomLayer.cpp` by walking through the code and making the necessary 
changes for the `CustomLayer` custom layer along the way.

1. Using the text editor, open the CPU extension source file `your/directory/cl_CustomLayer/user_ie_extensions/cpu/ext_CustomLayer.cpp`.

2. To implement the `CustomLayer` function to efficiently execute in parallel, the code will use the parallel processing supported by the Inference Engine through the use of the Intel® Threading Building Blocks library.  To use the library, at the top we must include the header [`ie_parallel.hpp`](https://docs.openvinotoolkit.org/2019_R3.1/ie__parallel_8hpp.html) file by adding the `#include` line as shown below.

    Before:

    ```cpp
    #include "ext_base.hpp"
    #include <cmath>
    ```

    After:

    ```cpp
    #include "ext_base.hpp"
    #include "ie_parallel.hpp"
    #include <cmath>
    ```

3. The class `CustomLayerImp` implements the `CustomLayer` custom layer and inherits from the extension layer base class `ExtLayerBase`.

    ```cpp
    class CustomLayerImpl: public ExtLayerBase {
        public:
    ```

4. The `CustomLayerImpl` constructor is passed the `layer` object that it is associated with to provide access to any layer parameters that may be needed when implementing the specific instance of the custom layer.

    ```cpp
    explicit CustomLayerImpl(const CNNLayer* layer) {
      try {
        ...
    ```

5. The `CustomLayerImpl` constructor configures the input and output data layout for the custom layer by calling `addConfig()`.  In the template file, the line is commented-out and we will replace it to indicate that `layer` uses `DataConfigurator(ConfLayout::PLN)` (plain or linear) data for both input and output.

    Before:

    ```cpp
    ...
    // addConfig({DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});

    ```

    After:

    ```cpp
    addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
    ```

6. The construct is now complete, catching and reporting certain exceptions that may have been thrown before exiting.

    ```cpp
      } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
      }
    }
    ```

7. The `execute` method is overridden to implement the functionality of the custom layer.  The `inputs` and `outputs` are the data buffers passed as [`Blob`](https://docs.openvinotoolkit.org/2019_R3.1/_docs_IE_DG_Memory_primitives.html) objects.  The template file will simply return `NOT_IMPLEMENTED` by default.  To calculate the custom layer, we will replace the `execute` method with the code needed to calculate the `CustomLayer` function in parallel using the [`parallel_for3d`](https://docs.openvinotoolkit.org/2019_R3.1/ie__parallel_8hpp.html) function.

    Before:

    ```cpp
      StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override {
        // Add here implementation for layer inference
        // Examples of implementations you can find in Inference Engine tool samples/extensions folder
        return NOT_IMPLEMENTED;
    ```

    After:
    ```cpp
      StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override {
        // Add implementation for layer inference here
        // Examples of implementations are in OpenVINO samples/extensions folder

        // Get pointers to source and destination buffers
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        // Get the dimensions from the input (output dimensions are the same)
        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        // Perform (in parallel) the hyperbolic cosine given by: 
        //    CustomLayer(x) = (e^x + e^-x)/2
        parallel_for3d(N, C, H, [&](int b, int c, int h) {
        // Fill output_sequences with -1
        for (size_t ii = 0; ii < b*c; ii++) {
          dst_data[ii] = (exp(src_data[ii]) + exp(-src_data[ii]))/2;
        }
      });
    return OK;
    }
    ```

##### Edit `CMakeLists.txt`

Because the implementation of the `CustomLayer` custom layer makes use of the parallel processing 
supported by the Inference Engine, we need to add the Intel® Threading Building Blocks 
dependency to `CMakeLists.txt` before compiling.  We will add paths to the header 
and library files and add the Intel® Threading Building Blocks library to the list of link libraries. 
We will also rename the `.so`.

1. Using the text editor, open the CPU extension CMake file `your/directory/cl_CustomLayer/user_ie_extensions/cpu/CMakeLists.txt`.
2. At the top, rename the `TARGET_NAME` so that the compiled library is named `libCustomLayer_cpu_extension.so`:

    Before:

    ```cmake
    set(TARGET_NAME "user_cpu_extension")
    ```

    After:
    
    ```cmake
    set(TARGET_NAME "CustomLayer_cpu_extension")
    ```

3. Now modify the `include_directories` to add the header include path for the Intel® Threading Building Blocks library located in `/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include`:

    Before:

    ```cmake
    include_directories (PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    )
    ```

    After:
    ```cmake
    include_directories (PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include"
    )
    ```

4. Now add the `link_directories` with the path to the Intel® Threading Building Blocks library binaries at `/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib`:

    Before:

    ```cmake
    ...
    #enable_omp()
    ```

    After:
    ```cmake
    ...
    link_directories(
    "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib"
    )
    #enable_omp()
    ```

5. Finally, add the Intel® Threading Building Blocks library `tbb` to the list of link libraries in `target_link_libraries`:

    Before:

    ```cmake
    target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
    ```

    After:

    ```cmake
    target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib} tbb)
    ```

#### Compile the Extension Library

To run the custom layer on the CPU during inference, the edited extension C++ source code 
must be compiled to create a `.so` shared library used by the Inference Engine. 
In the following steps, we will now compile the extension C++ library.

1. First, we run the following commands to use CMake to setup for compiling:

    ```bash
    cd your/directory/cl_CustomLayer/user_ie_extensions/cpu
    mkdir -p build
    cd build
    cmake ..
    ```

    The output will appear similar to:     

    ```
    -- Generating done
    -- Build files have been written to: your/directory/cl_tutorial/cl_CustomLayer/user_ie_extensions/cpu/build
    ```

2. The CPU extension library is now ready to be compiled.  Compile the library using the command:

    ```bash
    make -j $(nproc)
    ```

    The output will appear similar to: 

    ```
    [100%] Linking CXX shared library libCustomLayer_cpu_extension.so
    [100%] Built target CustomLayer_cpu_extension
    ```

### Execute the Model with the Custom Layer

#### Using a C++ Sample

To start on a C++ sample, we first need to build the C++ samples for use with the Inference
Engine:

```bash
cd /opt/intel/openvino/deployment_tools/inference_engine/samples/
./build_samples.sh
```

This will take a few minutes to compile all of the samples.

Next, we will try running the C++ sample without including the `CustomLayer` extension library to see 
the error describing the unsupported `CustomLayer` operation using the command:  

```bash
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -d CPU
```

The error output will be similar to:

```
[ ERROR ] Unsupported primitive of type: CustomLayer name: ModCustomLayer/CustomLayer/CustomLayer
```

We will now run the command again, this time with the `CustomLayer` extension library specified 
using the `-l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so` option 
in the command:

```bash
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -d CPU -l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so
```

The output will appear similar to:

```
Image /directory/path/pic.bmp

classid probability
------- -----------
0       0.9308984  
1       0.0691015

total inference time: xx.xxxxxxx
Average running time of one iteration: xx.xxxxxxx ms

Throughput: xx.xxxxxxx FPS

[ INFO ] Execution successful
```

#### Using a Python Sample

First, we will try running the Python sample without including the `CustomLayer` extension library 
to see the error describing the unsupported `CustomLayer` operation using the command:  

```bash
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -d CPU
```

The error output will be similar to:

```
[ INFO ] Loading network files:
your/directory/cl_tutorial/tf_model/model.ckpt.xml
your/directory/cl_tutorial/tf_model/model.ckpt.bin
[ ERROR ] Following layers are not supported by the plugin for specified device CPU:
ModCustomLayer/CustomLayer/CustomLayer, ModCustomLayer/CustomLayer_1/CustomLayer, ModCustomLayer/CustomLayer_2/CustomLayer
[ ERROR ] Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument
```

We will now run the command again, this time with the `CustomLayer` extension library specified 
using the `-l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so` option 
in the command:

```bash
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so -d CPU
```

The output will appear similar to:

```
Image your/directory/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

classid probability
------- -----------
0      0.9308984
1      0.0691015
```

### Why you might need to handle custom layers?

Some of the potential reasons for handling custom layers are:
- You might want to run some experimental layer on top of what already exists in the list of supported layer. 
- The Layers you're trying to run uses unsupported input/ output shapes or formats.
- You're trying to run a framework out of the support frameworks like Tensorflow, ONNX, Caffe.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End-User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
