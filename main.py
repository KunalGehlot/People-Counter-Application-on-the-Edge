"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# import numpy

# MQTT server environment variables

import os
import sys
import cv2
import time
import json
import socket
import logging as log
from inference import Network
from datetime import datetime
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS

# Time for a box to show up until not detected the next time
TIMEOUT = 2000

# List of boxes discovered in previously
PREVIOUS_BOXES = []

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser(
        "*-*-*-*-*-*-*-*-*-*-*-*    Run Inference with Video/ Image    *-*-*-*-*-*-*-*-*-*-*-*")

    parser.add_argument("-m", "--model", required=False, type=str,
                        help="Path to an xml file with a trained model.", default="frozen_inference_graph.xml")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file", default='resources/Pedestrian_Detect_2_1_1.mp4')
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    return parser


def draw_boxes(frame, result, width, height, prob_t):
    '''
    Draw bounding boxes onto the frame.
    '''

    color = (0,60,255)

    def confidence(box):
        return box[2]

    global PREVIOUS_BOXES
    boxes = []
    best_boxes = []
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_t and box[1] == 1:
            boxes.append(box)
    if len(boxes) > 0:
        boxes.sort(key=confidence, reverse=True)
        found = False
        best_boxes = []
        for box in boxes:
            if found == False:
                found = True
                best_boxes.append(box)
            else:
                append = True
                for best_box in best_boxes:
                    if not (box[3] > best_box[5] or box[5] < best_box[3] or box[4] > best_box[6] or box[6] < best_box[4]):
                        append = False
                        break
                if append == True:
                    best_boxes.append(box)

        PREVIOUS_BOXES = []
        for box in best_boxes:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            PREVIOUS_BOXES.append((datetime.timestamp(datetime.now()), box))
    else:
        
        if len(PREVIOUS_BOXES) > 0:
            for box in PREVIOUS_BOXES:
                if datetime.timestamp(datetime.now()) - box[0] < TIMEOUT and box[1][3] > 0.01 and box[1][5] < 0.99:
                    best_boxes.append(box[1])
            for box in best_boxes:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
    return frame, len(best_boxes)


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # print("**********\tinfer_on_stream initialized\t**********\n")
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    probabily_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device,
                             args.cpu_extension)

    in_shape = infer_network.get_input_shape()
    # print("----------\tInput Shape of the Model: " +
    #      str(in_shape), "\t----------")
    # exit(1)

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        # print("Unable to open input. Exiting...")
        exit(1)
    cap.open(args.input)
    # print("----------\tVideo Capture Opened\t----------")
#    exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("----------\tWidth:", width, "Height:", height, "\t----------")
    # exit(1)
    frames = 0
    found = False
    t_count = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # print("-----------\tStream Loop Started\t-----------")
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            # print("Cannot read the input stream. Exiting...")
            exit(1)
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        if frame_count == -1:
            frame = cv2.cvtColor((frame, cv2.COLOR_YUV2BGR_I420))
        p_frame = cv2.resize(frame, (in_shape[3], in_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        # print("----------\tImage Resized to fit: ",
        #      p_frame.shape, "\t----------")
        # exit(1)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        # print("----------\tASync Start\t----------")
        # cv2.imwrite("output0.jpg", frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            # print("----------\tASync Wait\t----------")

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            # print("----------\tInference Output: ",
            #      result.shape, "\t----------")

            ### TODO: Extract any desired stats from the results ###
            frame, count = draw_boxes(
                frame, result, width, height, probabily_threshold)
            # cv2.imwrite("output.jpg", frame)
            # exit(1)
            on_t = frames/fps
            ### Detect new person ###
            if not found and count > 0:
                t_count += count
                found = True
            if found and count > 0:
                frames += 1
            if found and count == 0:
                found = False
                ### Send to MQTT Server ###
                on_t = int(frames/fps)
                client.publish("person/duration",
                               json.dumps({"duration": on_t}))
                frames = 0
            ### Send to MQTT Server ###
            client.publish("person",
                           json.dumps({"count": count, "total": t_count}))

            # ### TODO: Extract any desired stats from the results ###
            on_t_mssg = "On Screen time: {:.3f}ms".format(on_t * 1000)
            count_mssg = "People counted: {}".format(t_count)
            # # print(on_t_mssg)
            # # print("----------\tOn Screen Time\t----------")
            # # print(count_mssg)
            # # print("----------\tTotal Count\t----------")
            # # exit(1)
            
            ### Write Scree-on time and count on screen ###
            cv2.putText(img=frame, text=str(count_mssg), org=(
                15, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(10, 60, 255), thickness=1)
            cv2.putText(img=frame, text=str(on_t_mssg), org=(
                15, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(10, 60, 255), thickness=1)
            # cv2.imwrite("output1.jpg", frame)
            # exit(1)
            
        ### TODO: Send the frame to the FFMPEG server ###
        if (frame_count > 0 or frame_count == -1):
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        else:
            cv2.imwrite("output.jpg", frame)
            # print("-*-*-*-*-*\tImage saved: output.jpg\t*-*-*-*-*-")
        if key_pressed == 27:
            break
        # exit(1)

    cap.release()
    cv2.destroyAllWindows()
    # TODO: Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    Network.kill()
    exit(0)
