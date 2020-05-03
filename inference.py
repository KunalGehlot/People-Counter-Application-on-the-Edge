#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
            ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None, plugin=None):
        print("**********\tNetwork.load_model initialized\t**********\n")
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        print("----------\tModel and Weights loaded\t----------")
        self.plugin = IECore()
        self.network = IENetwork(model = model_xml, weights = model_bin)
        print("----------\tIE Core and Network loaded\t----------")
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network = self.network, device_name = device)
        unsupported_layers = [
            l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore. Exiting...")
            exit(1)
        print("----------\tChecked for supported layers\t----------")
        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        print("----------\tAdded CPU extension\t----------")
        self.exec_network = self.plugin.load_network(self.network, device)
        print("----------\tNetwork loaded\t----------")
        it = iter(self.network.inputs)
        self.input_blob_info = None
        self.input_blob = next(it)
        if self.input_blob == "image_info":
            self.input_blob_info = self.input_blob
            self.input_blob = next(it)
        self.output_blob = next(iter(self.network.outputs))
        print("**********\tNetwork.load_model finished\t**********\n")
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. -- Loaded with model, device, CPU_EXTENSION, plugin ###


    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape
    
    def get_output_shape(self):
        ### NODO: Return the shape of the output layer ###
        return self.network.outputs[self.output_blob].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        inputs = {self.input_blob: image}
        if self.input_blob_info == "image_info":
            inputs = {self.input_blob_info: \
                (image.shape[2], image.shape[3], 1), self.input_blob: image}
        self.exec_network.start_async(request_id=0, inputs=inputs)
        ### TODO: Return any necessary information -- None ###
        ### Note: You may need to update the function parameters. -- Loaded with image ###

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].wait(-1)

    def get_output(self):
        # TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]

    def kill(self):
        print("----------   Killing Plugin and Network    ----------")
        del self.exec_network
        del self.plugin
        del self.network
