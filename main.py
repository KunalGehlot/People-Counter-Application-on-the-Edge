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


# MQTT server environment variables
import os
import sys
import cv2
import time
import json
import numpy
import socket
import logging as log
from inference import Network
import paho.mqtt.client as mqtt
from argparse import ArgumentParser

MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser("Run Inference with Video/ Image")

    parser.add_argument("-m", "--model", required=False, type=str,
                        help="Path to an xml file with a trained model.", default="mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml")
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
    count = 0
    for box in result[0][0]:  # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_t:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            count += 1
    return frame, count


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

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    probabily_threshold = args.prob_threshold
    ### TODO: Load the model through `infer_network` ###
    output = infer_network.load_model(args.model, args.device)
    # no, chnl = infer_network.load_model(
    #     args.model, args.device, args.cpu_extension)
    print("---------------------------------------  INFER NETWORK   ---------------------------------------")
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Unable to open input. Exiting...")
        exit(1)
    cap.open(args.input)
    print("---------------------------------------  CAP OPEN   ---------------------------------------")
#    exit(1)
    width = int(cap.get(3))
    height = int(cap.get(4))
    print(width, "---/---", height)
    print("---------------------------------------  WIDTH/HEIGHT   ---------------------------------------")
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        p_count = 0
        t_count = 0
        new_t = 0
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            print("Cannot read the input stream. Exiting...")
            exit(1)
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        old_size = frame.shape[:2] # old_size is in (height, width) format

        ratio = float(800)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        p_frame = cv2.resize(frame, (new_size[1], new_size[0]))

        delta_w = 800 - new_size[1]
        delta_h = 800 - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        p_frame = cv2.copyMakeBorder(p_frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        print(p_frame.shape, "\n---------------------------------------  FRAME   ---------------------------------------")
        p_frame = cv2.resize(p_frame, (800, 800))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, 3, 800, 800)
        print((p_frame.shape), "\n---------------------------------------  P_FRAME   ---------------------------------------")
        # exit(1)
        ### TODO: Start asynchronous inference for specified request ###
        start_t = time.time()
        
        print("---------------------------------------  MODEL LOADED!   ---------------------------------------")
        print("Output: ", output)
        handler = infer_network.exec_net(p_frame)
        print("---------------------------------------  Network executed   ---------------------------------------")
        ### TODO: Wait for the result ###
        if handler.wait() == 0:
            on_t = time.time() - start_t
        ### TODO: Get the results of the inference request ###
        result = infer_network.get_output()
        print("---------------------------------------  Output Blob Received!   ---------------------------------------")
        a,b,c,d = infer_network.get_input_shape()
        print("---------------------------------------  Got the shape output!   ---------------------------------------")
        frame, count = draw_boxes(
            frame, result, width, height, probabily_threshold)
        ### TODO: Extract any desired stats from the results ###
        on_t_mssg = "Screen on time: {:.3f}ms".format(on_t * 1000)
        
        ### Write Scree-on time and count on screen ###
        cv2.putText(frame, on_t_mssg, (15,15), 0.5, (215,20,20) ,1)
        cv2.putText(frame, on_t_mssg, (30,15), 0.5, (215,20,20) ,1)
        ### Detect new person ###
        if count > p_count:
            new_t = time.time()
            t_count += count - p_count
            ### Send to MQTT Server ###
            client.publish("Person", json.dumps({"Total Count": t_count}))
        
        if count < p_count:
            duration = int(time.time() - new_t)
            ### Send to MQTT Server ###
            client.publish("Person/ Duration", json.dumps({"Duration": duration}))
        ### Print count on screen ###
        count_mssg = "People counted: {0}".format(count)
        ### Send Count to MQTT Server ###
        client.publish("Person", json.dumps({"count": t_count}))
        p_count = count
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###

        if key_pressed == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    # TODO: Disconnect from MQTT
    client.disconnect()

    return


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    print(args, "\n---------------------------------------  ARGS    ---------------------------------------")
    
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    exit(0)
