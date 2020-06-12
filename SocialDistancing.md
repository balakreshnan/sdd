# Social Distiancing AI modelling

## Use Case

Based on the cornona virus situation recently in 2020 it has become apparent that social distancing enforcing is becoming common across industries where people gather. This could be any public or government or even corporate office spaces and manufacturing plants. Use case here is to detect person and identfying distance to see if they are maintaining distance between each other.

This article covers how to implement the solution using existing open source code that are available.

## Steps

- Download Yolo v3 or Yolo v3 tiny model (weights)
- Download Social distancing code
- Open Code using Visual Studio Code
- Run the sdd_gui.py
- Convert the code to deployment as iot edge
- deploy to edge device
- Conclusion

## Download Yolo V3

Go to this web site: https://pjreddie.com/darknet/yolo/
Download Yolo V3 around here - wget https://pjreddie.com/media/files/yolov3.weights
download Yolo V3 tiny around here - wget https://pjreddie.com/media/files/yolov3-tiny.weights
Create a folder where to download the code files and save the weights file there.

## Download Social Distancing Code

https://github.com/nadav01/sdd
I have forked from above as well as backup - https://github.com/balakreshnan/sdd
