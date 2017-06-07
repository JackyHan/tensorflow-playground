# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Used to generate input data points"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def classifyCircleData(numSamples):
    points = []
    radius = 5
    #getting the positive points inside the circle
    for i in range(0, numSamples//2):
        r = randUniform(0, radius * 0.5)
        angle = randUniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        points.append([x,y,0])
    # getting the negative points outside the circle
    for i in range(0, numSamples//2):
        r = randUniform(radius * 0.7, radius)
        angle = randUniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        points.append([x,y,1])
    points = np.array(points, dtype=np.float32)
    # return np.random.permutation(points)
    return points
     

def classifyGaussData(numSamples):
    """ getting a specific number of gauss data points
    i.e. simple linear data"""
    points = []
    radius = 6
    for i in range(0, numSamples//2):
        x = np.random.normal(-radius//2, 1)
        y = np.random.normal(-radius//2, 1)
        if(x+y>=0):
            points.append([x,y,0])
        else:
            points.append([x,y,1])
    for i in range(0, numSamples//2):
        x = np.random.normal(radius//2, 1)
        y = np.random.normal(radius//2, 1)
        if(x+y>=0):
            points.append([x,y,0])
        else:
            points.append([x,y,1])
    points = np.array(points, dtype=np.float32)
    # return np.random.permutation(points)
    return points
    
def classifyXORData(numSamples):
    points = []
    radius = 5
    padding = 0.3;
    for i in range(0, numSamples):
        x = randUniform(-radius,radius)
        if x > 0:
            x+=padding
        else:
            x+=-padding
        y = randUniform(-radius,radius)
        if y > 0:
            y+=padding
        else:
            y+=-padding
        if(x*y>=0):
            points.append([x,y,0])
        else:
            points.append([x,y,1])
    points = np.array(points, dtype=np.float32)
    # return np.random.permutation(points)
    return points

def classifySpiralData(numSamples):
    points = []
    n = numSamples//2
    # first class with label 0
    for i in range(0, n):
        r = float(i)/ n * 5
        t = 1.75 * i / n * 2 * math.pi + 0
        x = r * math.sin(t)
        y = r * math.cos(t)
        points.append([x,y,0])
    # second class with label 1
    for i in range(0, n):
        r = float(i) / n * 5
        t = 1.75 * i / n * 2 * math.pi + math.pi
        x = r * math.sin(t)
        y = r * math.cos(t)
        points.append([x,y,1])
    points = np.array(points, dtype=np.float32)
    # return np.random.permutation(points)
    return points
    
def randUniform(a, b):
    """ return a random number between a and b"""

    return random.uniform(0, 1) * (b - a) + a
    
    
def main(argv=None):  # pylint: disable=unused-argument
    points = classifySpiralData(500)
    for i in range(0, 250):
        plt.scatter(x=points[i,0], y=points[i,1], c = '#0800fc')
    for i in range(250, 500):
        plt.scatter(x=points[i,0], y=points[i,1], c = '#fc9f00')
    plt.axis([-6, 6, -6, 6])
    plt.show()
    

if __name__ == '__main__':
    main()


