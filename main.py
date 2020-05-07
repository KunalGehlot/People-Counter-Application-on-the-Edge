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
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_t:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count


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

    imageFlag = False

    previous = 0
    total = 0
    start = 0

    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device,
                             args.cpu_extension)
    
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        imageFlag = True
        input_stream = args.input
    else:
        input_stream = args.input
    # print("----------\tInput Shape of the Model: " +
    #      str(in_shape), "\t----------")
    # exit(1)

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        print("Unable to open input. Exiting...")
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

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # print("-----------\tStream Loop Started\t-----------")
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            print("Cannot read the input stream. Exiting...")
            exit(1)
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame,(w,h))
        image = image.transpose((2,0,1))
        image = image.reshape((n. c. h. w))
        # print("----------\tImage Resized to fit: ",
        #      p_frame.shape, "\t----------")
        # exit(1)

        start_t = time.time()
        infer_network.exec_net(0, image)
        ### TODO: Start asynchronous inference for specified request ###
        # print("----------\tASync Start\t----------")
        # cv2.imwrite("output0.jpg", frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            # print("----------\tASync Wait\t----------")
            d_time = time.time() - start_t
            ### TODO: Get the results of the inference request ###
        result = infer_network.get_output(0)
            # print("----------\tInference Output: ",
            #      result.shape, "\t----------")

            ### TODO: Extract any desired stats from the results ###
        frame, count = draw_boxes(
            frame, result, width, height, probabily_threshold)
            # cv2.imwrite("output.jpg", frame)
            # exit(1)
        if count > previous:
            start = time.time()
            total+= count - previous
            client.publish("person", json.dumps({"total": total}))

            # Person duration in the video is calculated
        if count < previous:
            duration = int(time.time() - start)
               # Publish messages to the MQTT server
            client.publish("person/duration",
                              json.dumps({"duration": duration}))

       
        client.publish("person", json.dumps({"count": count}))
        previous = count      

            # ### TODO: Extract any desired stats from the results ###
        d_time_mssg = "On Screen time: {:.3f}ms".format(d_time * 1000)
        count_mssg = "People counted: {}".format(total)
            # # print(on_t_mssg)
            # # print("----------\tOn Screen Time\t----------")
            # # print(count_mssg)
            # # print("----------\tTotal Count\t----------")
            # # exit(1)

            ### Write Scree-on time and count on screen ###
        cv2.putText(img=frame, text=str(count_mssg), org=(
            15, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(10, 60, 255), thickness=1)
        cv2.putText(img=frame, text=str(d_time_mssg), org=(
            15, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(10, 60, 255), thickness=1)
            # cv2.imwrite("output1.jpg", frame)
            # exit(1)

        if key_pressed == 10:
            cv2.imwrite("output.jpg", frame)

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
