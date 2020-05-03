# People Counter (AI on the Edge)

### This Project is a submission of the first project in **Intel Edge AI Nanodegree** at Udacity.

# Index

- [Introduction](#Introduction)
  -
  - What it Does
  - How it Works
  - [Requirements](#requirements-only-for-this-code)
    - Hardware
    - Software

- [How to Run](#how-to-run)
  -
  - Step 1 - Start the Mosca server
  - Step 2 - Start the GUI
  - Step 3 - FFmpeg Server
  - Step 4 - Run the code
    - [Running on CPU](#running-on-cpu-default)
- [Project Write-Up](#project-write-up)
  - 
  - [What are Custom Layers](#what-are-custom-layers)
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

# Project Write-Up

## What are Custom Layers

OpenVINO Toolkit Documentations has a [list of Supported Framework Layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) for DL Inference. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

*(Model Optimizer is a cross-platform command-line tool that facilitates the transition between the training and deployment environment, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices. [Learn more about it here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).)*



Some of the potential reasons for handling custom layers are...

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
