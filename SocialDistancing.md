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
The above code uses Yolo models to detect person and then do calculation to create alerts.

Huge thanksa to Nadav Leobl and Lorea Arrizabalaga

Make sure Code and Yolo weights and config files are in the same folder.

## Open the Code to Run

Now open Visual Studio Code and open the folder.
```
Note: Conda environment has to be setup and necessary libraries have to be installed. If necessary please do so.
```
Wait until all dependency check are all completed.
Now on the right top corner you should see a play button and press that and see if it works.

## Run sdd_gui.py Code

You should see a popup screen with start button.
Just press the start button to invoke the laptop camera.
Test the model and see if works.

## Convert the code to deployment as iot edge

## deploy to edge device

## Conclusion