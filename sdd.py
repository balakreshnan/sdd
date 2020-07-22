import cv2, numpy
from twilio.rest import Client
from playsound import playsound
from azure.eventhub import EventHubClient, Sender, EventData
import json
import requests
from datetime import datetime
import argparse
import tensorflow as tf
import numpy as np
import PIL.Image

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

confid, threshold = 0.5, 0.5

ADDRESS = "amqps://babalingest.servicebus.windows.net/sddinput"
USER = "ingest"
KEY = "6Sur+IoqeH69ETumOzo0wh5V/L1FUOdpH0yzhK5cqWg="

class ObjectDetection:
    INPUT_TENSOR_NAME = 'image_tensor:0'
    OUTPUT_TENSOR_NAMES = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']

    def __init__(self, model_filename):
        graph_def = tf.compat.v1.GraphDef()
        with open(model_filename, 'rb') as f:
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Get input shape
        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.input_shape = sess.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME).shape.as_list()[1:3]

    def predict_image(self, image):
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = image.resize(self.input_shape)

        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensors = [sess.graph.get_tensor_by_name(n) for n in self.OUTPUT_TENSOR_NAMES]
            outputs = sess.run(output_tensors, {self.INPUT_TENSOR_NAME: inputs})
            return outputs

def predict(model_filename, image_filename):
    od_model = ObjectDetection(model_filename)

    image = PIL.Image.open(image_filename)
    return od_model.predict_image(image)

def predictframe(model_filename, frame):
    od_model = ObjectDetection(model_filename)

    image = PIL.Image.open(image_filename)
    return od_model.predict_image(image)


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def orgdist(p1, p2):
    d = dist(p1[2], p2[2])
    w = (p1[0] + p2[0]) / 2
    h = (p1[1] + p2[1]) / 2
    _ = 0
    try:
        _ = (p2[2][1] - p1[2][1]) / (p2[2][0] - p1[2][0])
    except ZeroDivisionError:
        _ = 1.633123935319537e+16
        #print("Only 1 Person detected")
    #print("Distance calculated: " + str(_))
    return str(_)

def isclose(p1, p2):
    d = dist(p1[2], p2[2])
    w = (p1[0] + p2[0]) / 2
    h = (p1[1] + p2[1]) / 2
    _ = 0
    try:
        _ = (p2[2][1] - p1[2][1]) / (p2[2][0] - p1[2][0])
    except ZeroDivisionError:
        _ = 1.633123935319537e+16
        #print("Only 1 Person detected")
    ve = abs(_ / ((1 + _ ** 2) ** 0.5))
    ho = abs(1 / ((1 + _ ** 2) ** 0.5))
    d_hor = ho * d
    d_ver = ve * d
    vc_calib_hor = w * 1.3
    vc_calib_ver = h * 0.4 * 0.8 # the last one is the angle
    c_calib_hor = w * 1.7
    c_calib_ver = h * 0.2 * 0.8 # the last one is the angle
    #print("Distance calculated: " + str(_))
    if 0 < d_hor < vc_calib_hor and 0 < d_ver < vc_calib_ver:
        return 1
    elif 0 < d_hor < c_calib_hor and 0 < d_ver < c_calib_ver:
        return 2
    else:
        return 0


def run(camera='webcam', sound=False, sms=''):
    # Twilio config
    twilio_account_sid = ''
    twilio_auth_token = ''
    twilio_phone_number = ''
    destination_phone_number = ''
    client = None
    sms_sent = False
    sms_timer = 0
    sms_limit = 50 # send sms every sms_limit frames at most

    if sms != '':
        twilio_account_sid = ''
        twilio_auth_token = ''
        twilio_phone_number = ''
        destination_phone_number = '+972' + sms[1:]
        client = Client(twilio_account_sid, twilio_auth_token)


    LABELS = open("./coco.names").read().strip().split("\n")
    numpy.random.seed(42)

    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"
    #weightsPath = "./yolov5s.pt"
    #configPath = "./yolov5s.yaml"

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    FR = 0
    vs = None

    #send_url = 'http://freegeoip.net/json'
    #r = requests.get(send_url)
    #j = json.loads(r.text)
    #lat = j['latitude']
    #lon = j['longitude']
    lat = "42.8895"
    lon = "-87.9074"

    if camera == 'webcam':
        vs = cv2.VideoCapture(0)

    else:
        vs = cv2.VideoCapture(camera)

    writer = None
    (W, H) = (None, None)

    fl = 0
    q = 0
    while True:

        # Hit 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            FW = W
            if(W<1075):
                FW = 1075
            FR = numpy.zeros((H+210,FW,3), numpy.uint8)
            col = (255,255,255)
            FH = H + 210
        FR[:] = col

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        label = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = numpy.argmax(scores)
                confidence = scores[classID]
                if LABELS[classID] == "person":
                    if confidence > confid:
                        box = detection[0:4] * numpy.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        label.append(LABELS[classID])
                        #print("Boxes: " + str(box))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, threshold)

        if len(idxs) > 0:
            status = []
            idf = idxs.flatten()
            close_pair = []
            s_close_pair = []
            center = []
            co_info = []
            distance1 = 0

            for i in idf:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cen = [int(x + w / 2), int(y + h / 2)]
                center.append(cen)
                cv2.circle(frame, tuple(cen), 1, (0,0,0), 1)
                co_info.append([w, h, cen])
                status.append(0)

            #print("Center: " + str(center))

            for i in range(len(center)):
                for j in range(len(center)):
                    ans = isclose(co_info[i], co_info[j])
                    distance = orgdist(co_info[i], co_info[j])

                    if dist(center[i], center[j]) > 0:
                        distance1 = dist(center[i], center[j])
                        #print("Distance calculated: " + str(distance1))

                    if ans == 1:
                        close_pair.append([center[i], center[j]])
                        status[i] = 1
                        status[j] = 1
                    elif ans == 2:
                        s_close_pair.append([center[i], center[j]])
                        if status[i] != 1:
                            status[i] = 2
                        if status[j] != 1:
                            status[j] = 2

                        if sms != '' and not sms_sent:
                            client.messages.create(
                                body="There has been social distancing breach",
                                from_=twilio_phone_number,
                                to=destination_phone_number)
                            sms_sent = True
                            sms_timer = 0

                        if sound:
                            playsound('alarm.mp3', block=False)
                    
            #print(" validated distance: " + str(ans))
            #print(" validated distance: " + str(ans) + " Social Distance Answer: " + str(distance) + " value i: " + str(center[0]) + " len center: " + str(len(center[0])))
            #print("Layered output: " + str(layerOutputs))
            #print("Layered output: " + str(len(layerOutputs)))
            #print(" Labels: " + str(label))

            imageWidth  = frame.shape[0]
            imageHeight = frame.shape[1]
            array       = frame[6]
            image_string = str(bytearray(frame))

            #print("imge height " + str(imageHeight) + " image widht: " + str(imageWidth))
            print("Frame shape: " + str(frame.shape))
            # Create a PIL Image from our pixel array.
            #im = PIL.Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
            #posepredict = predict_image(im)
            #print("Pose: " + str(posepredict))
            #image = im.resize(self.input_shape)

            INPUT_TENSOR_NAME = 'image_tensor:0'
            OUTPUT_TENSOR_NAMES = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']
            model_filename = 'model.pb'
            labels_filename = 'labels.txt'

            posepredict = "";
            posepredictprob = 0;

            dim = (320, 320)

            reframe = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            graph_def = tf.compat.v1.GraphDef()
            with open(model_filename, 'rb') as f:
                graph_def.ParseFromString(f.read())

            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')

            # Get input shape
            with tf.compat.v1.Session(graph=graph) as sess:
                sess.input_shape = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME).shape.as_list()[1:3]

            inputs = np.array(reframe, dtype=np.float32)[np.newaxis, :, :, :]
            with tf.compat.v1.Session(graph=graph) as sess:
                output_tensors = [sess.graph.get_tensor_by_name(n) for n in OUTPUT_TENSOR_NAMES]
                outputs = sess.run(output_tensors, {INPUT_TENSOR_NAME: inputs})
                #print("output " + str(outputs))

            with open(labels_filename) as f:
                labels = [l.strip() for l in f.readlines()]

            for pred in zip(*outputs):
                #print(f"Class: {labels[pred[2]]}, Probability: {pred[1]}, Bounding box: {pred[0]}")
                posepredict = labels[pred[2]]
                posepredictprob = pred[1]

            total_p = len(center)
            low_risk_p = status.count(2)
            high_risk_p = status.count(1)
            safe_p = status.count(0)
            i = 0

            #message = str(label[0]) + "," + str(distance) + "," + str(center[0]) + "," + str(len(center[0])) + "," + str(status.count(1))

            #message = { """label""" + ": " + str(label[0]) + ", ""distance""" + ":" + str(distance) + ", ""center""" + ":" + str(center[0]) + ", ""length""" + ": " + str(len(center[0])) + ", ""highrisk""" + ":" + str(status.count(1)) }

            data = {}
            data['label'] = str(label[0])
            data['distance'] = str(distance1)
            data['center'] = str(center)
            data['length'] = str(len(center))
            data['highrisk'] = str(status.count(1))
            data['ans'] = str(ans)
            data['close_pair'] = str(close_pair)
            data['s_close_pair'] = str(s_close_pair)
            data['lowrisk'] = str(status.count(2))
            data['safe_p'] = str(status.count(0))
            data['total_p'] = str(total_p)
            data['lat'] = str(lat)
            data['lon'] = str(lon)
            data['serialno'] = "hack20201"
            #data['eventtime'] = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
            data['eventtime'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            data['posepredict'] = posepredict
            data['posepredictprob'] = str(posepredictprob)

            
            print("Message: " + json.dumps(data))
            client = EventHubClient(ADDRESS, debug=False, username=USER, password=KEY)
            sender = client.add_sender(partition="0")
            client.run()

            sender.send(EventData(json.dumps(data)))

            client.stop()

            

            for i in idf:
                tot_str = "Number of People: " + str(total_p)
                high_str = "High Risk: " + str(high_risk_p)
                low_str = "Low Risk: " + str(low_risk_p)
                safe_str = "Safe: " + str(safe_p)
                SDD = "Social Distancing Detector For Employee Well Being"

                cv2.putText(FR, tot_str, (20, H +25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(FR, 'Press q number of times to exit', (20, H + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(FR, safe_str, (260, H +25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 0), 2)
                cv2.putText(FR, low_str, (370, H +25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
                cv2.putText(FR, high_str, (520, H +25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
                cv2.putText(FR, SDD, (820, H + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if status[i] == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

                elif status[i] == 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

                i += 1

            for h in close_pair:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)

            for b in s_close_pair:
                cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

            FR[0:H, 0:W] = frame
            frame = FR
            cv2.imshow('Social Distancing Detector', frame)
            cv2.waitKey(1)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("op_", fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        writer.write(frame)
        sms_timer += 1
        if sms_timer > sms_limit:
            sms_sent = False

    vs.release()
