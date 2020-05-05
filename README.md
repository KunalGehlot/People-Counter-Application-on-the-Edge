# People Counter (AI on the Edge)<!-- omit in toc -->

### This Project is a submission of the first project in **Intel Edge AI Nanodegree** at Udacity.<!-- omit in toc -->

Find the Official Project Write-up [here](Writeup.md)

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
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -d CPU -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

*(Note: This command is only for running inference on CPU in Linux environment with Faster R-CNN Inception V2 on the given video file in resources, for any changes refer to default project instructions.)*

#### Viewing the App in your Browser

To see the output on a web based interface, open the link http://0.0.0.0:3004 in a browser.

*(Note: If the above URL Redirects to Intel's Github, try visiting http://0.0.0.0:3000 or redirecting the stream to Port 3005.)*