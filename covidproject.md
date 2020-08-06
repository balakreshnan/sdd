# Employee Well Being - Covid workplace safety analytics system

## Building systems that can protect employees when they return to work

```
Note: This project is open sourced. There is no intention to use person picture for any other use. We take privacy and security very serious. Other's when you use the project be cogizant of the privacy and security please.
```

## Use Case

Given the Covid times, we would like to build a system that uses artificial intelligence or machine learning or deep learning to identify people, mask and social distancing. The system should be able to report back what objects it was abel to detect. At the time of object being detected the system should also calculate 

- distance
- object detected
- center found
- how many centers found
- close pair - violated list
- count of how many where risk
- total person found
- Latitude
- Longitude
- Serial Number
- Event time

Above collected data is collected and stored for long term storage and also queryable storage to report. Reports are build to display KPI's.

- Total Violation per day - high risk
- Average people at the time of violation
- Average distance at the time of violation
- Time chart when it was detected for that day
- Time chart for week/month to show the trend on how the violation.
- Map of devices and show count of violation on a map - to cover different locations.
- Drill down based on locations for above KPI.

## Whiteboard - Architectural Design Session Outcome

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/whiteboard1.png "Whiteboard")

## Architecture

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/hack2020-2.jpg "Architecture")

## Architecture Explained.

## Steps

- Model Building <br/>
    Use Existing open source yolo v3 model <br/>
    Calculate social Distance <br/>
    Build Model to detect Mask <br/>
- Collecting Event data and send to Event hub
- Stream Analytics to process data from Event Hub
- Store data in ADLS GEN2 for long term storage and anlaytics
- Store Azure SQL Database for KPI reporting

## Sequence diagram

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/flow1.jpg "Flow")

## Logic Flow diagram

For Social Distancing detector <br/>

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/socialDistDetector.jpg "Social Distancing")

For Reporting and Cloud processing <br/>

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/Reporting.jpg "Reporting")

## Model Building

We are using python to build the model inferencing code. We are also leveraging yolo v3 tiny version.

```
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

Configuration

https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg

## Code

sdd_gui.py is the main visual program to start the video.

sdd.py is the code which inference and does the social distancing calculation and also packages up the event and send's to event hub for further process.

Sample event looks liks

```
{"label": "person", "distance": "265.41476974727686", "center": "[[580, 345], [576, 346], [317, 288]]", "length": "3", "highrisk": "3", "ans": "0", "close_pair": "[[[580, 345], [576, 346]], [[580, 345], [317, 288]], [[576, 346], [580, 345]], [[576, 346], [317, 288]], [[317, 288], [580, 
345]], [[317, 288], [576, 346]]]", "s_close_pair": "[]", "lowrisk": "0", "safe_p": "0", "total_p": "3", "lat": "42.8895", "lon": "-87.9074", "serialno": "hack20201", "eventtime": "22/07/2020 07:23:19"}
Message: {"label": "person", "distance": "268.68941177500835", "center": "[[583, 343], [579, 343], [316, 288]]", "length": "3", "highrisk": "3", "ans": "0", "close_pair": "[[[583, 343], [316, 288]], [[579, 343], [316, 288]], [[316, 288], [583, 343]], [[316, 288], [579, 343]]]", "s_close_pair": "[]", "lowrisk": "0", "safe_p": "0", "total_p": "3", "lat": "42.8895", "lon": "-87.9074", "serialno": "hack20201", "eventtime": "22/07/2020 07:23:21"}
```

Run the sdd_gui.py. 

```
Note: the application needs conda environment to run tensorflow models. Also the yolo and weights are needed.
```

Click Start button with out filling anything for local web camera

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/start1.jpg "Risk")

Wait for the Video screen with boxes to show up.

Check the output window to see messages that are sent to event hub.

Risk Profile

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/risk1.jpg "Risk")

## Process Events from Event hub

To process events form event hub we are using stream analytics to read the events from event hub and write to 2 outputs here

- Azure data lake store gen2 for long term storage
- Azure SQL database for reporting

## Azure Data Lake Store 

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/adlsgen21.jpg "ADLS gen 2")

## Azure SQL Databse

SQL Schema

```
create table dbo.sdddeetails
(
    label varchar(50),
    distance varchar(100),
    center varchar(100),
    length varchar(50),
    highrisk varchar(50),
    ans varchar(50),
    close_pair varchar(200),
    s_close_pair varchar(200),
    lowrisk varchar(50),
    safe_p varchar(50),
    total_p varchar(50),
    lat varchar(50),
    lon varchar(50),
    serialno varchar(50),
    posepredict varchar(50,
    posepredictprob varchar(50),
    maskPredict varchar(50,
    maskPredictProb varchar(50),
    EventProcessedUtcTime datetime,
    EventEnqueuedUtcTime datetime
)
```

Let's add more details to the modeling to hold company, locations and serial no.

```
create table company
(
    companyid bigint IDENTITY(1,1),
    companyname varchar(250),
    address1 varchar(250),
    address2 varchar(250),
    city varchar(100),
    state varchar(50),
    zip varchar(15),
    country varchar(250),
    parent varchar(1),
    parentid bigint,
    inserttime datetime
)

insert into company(companyname, address1,city,state,zip,country,parent,parentid,inserttime) values('company A', '850 east address street','city','WA','xxxxx','US','n',0,getdate())

create table locations
(
    locationid bigint IDENTITY(1,1),
    companyid bigint,
    locationname varchar(200),
    address1 varchar(250),
    address2 varchar(250),
    city varchar(100),
    state varchar(50),
    zip varchar(15),
    country varchar(250),
    latitude varchar(25),
    longitude varchar(25),
    active varchar(1),
    inserttime datetime
)

insert into locations(companyid,locationname,address1,city,state,zip,country,latitude,longitude,active,inserttime) values(1,'city','1 dre ave','xyz','WI','12345','US','xxxxxxxx','xxxxxxxx','1',getdate())


create table serialno
(
    id bigint IDENTITY(1,1),
    locationid bigint,
    serialno varchar(100),
    latitude varchar(25),
    longitude varchar(25),
    active varchar(1),
    inserttime datetime
)

insert into serialno(locationid,serialno,latitude,longitude,active,inserttime) values (1,'hack20201','xxxxxxx','xxxxxx','1',getdate())

create table violationaggr
(
SerialNo varchar(50),
Label varchar(50),
Avgdistance float,
Maxdistance float,
Mindistance float,
Countofhighrisk float,
Avghighrisk float,
Countsafep float,
Avgsafep float,
Counttotalp float,
Avgtotalp float,
StartTime datetime,
EndTime datetime
)
```

Now let's build the SQL data store to store all data and do some data modelling.

```
Note the above company, location tables are sample. for proper implementation please use below data model.
```

Here is the ER diagram for more meta data for the above data we are collecting.

To make it a real system we decided to add more details.

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/DatabaseERDiagram.jpg "ER diagram")

so for above ER diagram the SQL are 

```
/****** Object:  Table [dbo].[BuildingDevices]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BuildingDevices](
	[BuildingDeviceId] [int] IDENTITY(1,1) NOT NULL,
	[RoomId] [int] NULL,
	[DeviceId] [int] NULL,
 CONSTRAINT [PK__Building__11661996594571B7] PRIMARY KEY CLUSTERED 
(
	[BuildingDeviceId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BuildingRooms]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BuildingRooms](
	[RoomId] [int] IDENTITY(1,1) NOT NULL,
	[BuildingId] [int] NULL,
	[FloorNumber] [int] NULL,
	[Wing] [varchar](400) NULL,
	[RoomName] [varchar](800) NULL,
	[RoomDescription] [varchar](800) NULL,
 CONSTRAINT [PK__Building__328639394CD8D22D] PRIMARY KEY CLUSTERED 
(
	[RoomId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Buildings]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Buildings](
	[BuildingId] [int] IDENTITY(1,1) NOT NULL,
	[LocationId] [int] NULL,
	[BuildingName] [varchar](800) NULL,
	[BuildingDescription] [varchar](8000) NULL,
 CONSTRAINT [PK__Building__5463CDC4D2ECF49E] PRIMARY KEY CLUSTERED 
(
	[BuildingId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Company]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Company](
	[CompanyId] [int] IDENTITY(1,1) NOT NULL,
	[companyname] [varchar](800) NULL,
	[address1] [varchar](8000) NULL,
	[address2] [varchar](8000) NULL,
	[city] [varchar](400) NULL,
	[state] [varchar](400) NULL,
	[zip] [varchar](100) NULL,
	[country] [varchar](200) NULL,
	[parent] [int] NULL,
	[parentid] [int] NULL,
	[inserttime] [datetime] NULL,
 CONSTRAINT [PK__Company__AD5755B8E30FA13C] PRIMARY KEY CLUSTERED 
(
	[CompanyId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Devices]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Devices](
	[DeviceId] [int] IDENTITY(1,1) NOT NULL,
	[DeviceName] [varchar](800) NULL,
	[DeviceType] [varchar](800) NULL,
	[DeviceSerialNumber] [varchar](50) NOT NULL,
 CONSTRAINT [PK__Devices__49E1231141A059F5] PRIMARY KEY CLUSTERED 
(
	[DeviceId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_DeviceSerialNumber] UNIQUE NONCLUSTERED 
(
	[DeviceSerialNumber] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Location]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Location](
	[LocationId] [int] IDENTITY(1,1) NOT NULL,
	[CompanyId] [int] NULL,
	[address1] [varchar](8000) NULL,
	[address2] [varchar](8000) NULL,
	[city] [varchar](400) NULL,
	[state] [varchar](400) NULL,
	[zip] [varchar](100) NULL,
	[Latitude] [varchar](80) NULL,
	[Longitude] [varchar](80) NULL,
 CONSTRAINT [PK__Location__306F78A6EB400AE6] PRIMARY KEY CLUSTERED 
(
	[LocationId] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[sdddeetails]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[sdddeetails](
	[label] [varchar](50) NULL,
	[distance] [varchar](100) NULL,
	[center] [varchar](100) NULL,
	[length] [varchar](50) NULL,
	[highrisk] [varchar](50) NULL,
	[ans] [varchar](50) NULL,
	[close_pair] [varchar](200) NULL,
	[s_close_pair] [varchar](200) NULL,
	[lowrisk] [varchar](50) NULL,
	[safe_p] [varchar](50) NULL,
	[total_p] [varchar](50) NULL,
	[lat] [varchar](50) NULL,
	[lon] [varchar](50) NULL,
	[serialno] [varchar](50) NULL,
	[EventProcessedUtcTime] [datetime] NULL,
	[EventEnqueuedUtcTime] [datetime] NULL,
	[eventtime] [datetime] NULL,
	[posepredict] [varchar](50) NULL,
	[posepredictprob] [varchar](50) NULL,
	[maskPredict] [varchar](50) NULL,
	[maskPredictProb] [varchar](50) NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[violationaggr]    Script Date: 7/27/2020 10:30:56 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[violationaggr](
	[SerialNo] [varchar](50) NULL,
	[Label] [varchar](50) NULL,
	[Avgdistance] [float] NULL,
	[Maxdistance] [float] NULL,
	[Mindistance] [float] NULL,
	[Countofhighrisk] [float] NULL,
	[Avghighrisk] [float] NULL,
	[Countsafep] [float] NULL,
	[Avgsafep] [float] NULL,
	[Counttotalp] [float] NULL,
	[Avgtotalp] [float] NULL,
	[StartTime] [datetime] NULL,
	[EndTime] [datetime] NULL
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[BuildingDevices]  WITH CHECK ADD  CONSTRAINT [FK_BuildingDevices_BuildingRooms] FOREIGN KEY([RoomId])
REFERENCES [dbo].[BuildingRooms] ([RoomId])
GO
ALTER TABLE [dbo].[BuildingDevices] CHECK CONSTRAINT [FK_BuildingDevices_BuildingRooms]
GO
ALTER TABLE [dbo].[BuildingDevices]  WITH CHECK ADD  CONSTRAINT [FK_BuildingDevices_Devices] FOREIGN KEY([DeviceId])
REFERENCES [dbo].[Devices] ([DeviceId])
GO
ALTER TABLE [dbo].[BuildingDevices] CHECK CONSTRAINT [FK_BuildingDevices_Devices]
GO
ALTER TABLE [dbo].[BuildingRooms]  WITH CHECK ADD  CONSTRAINT [FK_BuildingRooms_Buildings] FOREIGN KEY([BuildingId])
REFERENCES [dbo].[Buildings] ([BuildingId])
GO
ALTER TABLE [dbo].[BuildingRooms] CHECK CONSTRAINT [FK_BuildingRooms_Buildings]
GO
ALTER TABLE [dbo].[Buildings]  WITH CHECK ADD  CONSTRAINT [FK_Buildings_Location] FOREIGN KEY([LocationId])
REFERENCES [dbo].[Location] ([LocationId])
GO
ALTER TABLE [dbo].[Buildings] CHECK CONSTRAINT [FK_Buildings_Location]
GO
ALTER TABLE [dbo].[Location]  WITH CHECK ADD  CONSTRAINT [FK_Location_Company] FOREIGN KEY([CompanyId])
REFERENCES [dbo].[Company] ([CompanyId])
GO
ALTER TABLE [dbo].[Location] CHECK CONSTRAINT [FK_Location_Company]
GO
ALTER TABLE [dbo].[sdddeetails]  WITH CHECK ADD  CONSTRAINT [FK_sdddeetails_SerialNumber] FOREIGN KEY([serialno])
REFERENCES [dbo].[Devices] ([DeviceSerialNumber])
ON UPDATE CASCADE
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[sdddeetails] CHECK CONSTRAINT [FK_sdddeetails_SerialNumber]
GO
ALTER TABLE [dbo].[violationaggr]  WITH CHECK ADD  CONSTRAINT [FK_violationaggr_SerialNumber] FOREIGN KEY([SerialNo])
REFERENCES [dbo].[Devices] ([DeviceSerialNumber])
ON UPDATE CASCADE
ON DELETE CASCADE
GO
ALTER TABLE [dbo].[violationaggr] CHECK CONSTRAINT [FK_violationaggr_SerialNumber]
GO
```

## Azure Stream Analytics

Since we are writing to multiple locations, we are creating a CTE to hold the temp data and then write to multiple location.

We can also do windowing, anamoly and other aggregation in stream analytics and persist to table also.

stream analytics has 2 inputs and 4 outputs

inputs

```
input - Event hub input for device data
sqlreferenceinput - Azure SQL based Reference data for power bi realtime
```

SQL input Reference Data query

```
select address1,city,state,b.BuildingName,Latitude,Longitude,FloorNumber, RoomName, DeviceSerialNumber  from Location a
inner join Buildings b
on a.LocationId = b.LocationId
inner join BuildingRooms c
on b.BuildingId = c.BuildingId
inner join BuildingDevices d
on d.RoomId = c.RoomId
inner join Devices e
on e.DeviceId = d.DeviceId
```

Outputs

```
outputblob - Output for ADLS gen2 storage for long term
aggrsqloutput - Aggregated output every 1 minute to Azure SQL database table
sqloutput - Raw data parsed and stored in Azure SQL database table for power BI report
pbioutput - Streaming data set to Power BI
```

```
WITH sddinput AS
(
    SELECT
    label,
    distance,
    center,
    length,
    highrisk,
    ans,
    close_pair,
    s_close_pair,
    lowrisk,
    safe_p,
    total_p,
    lat,
    lon,
    serialno,
    eventtime,
    posepredict,
    posepredictprob,
    maskPredict,
    maskPredictProb,
    EventProcessedUtcTime,
    EventEnqueuedUtcTime
    FROM input
)
SELECT
    label,
    distance,
    center,
    length,
    highrisk,
    ans,
    close_pair,
    s_close_pair,
    lowrisk,
    safe_p,
    total_p,
    lat,
    lon,
    serialno,
    eventtime,
    posepredict,
    posepredictprob,
    maskPredict,
    maskPredictProb,
    EventProcessedUtcTime,
    EventEnqueuedUtcTime
INTO outputblob
FROM sddinput

SELECT
    label,
    distance,
    center,
    length,
    highrisk,
    ans,
    close_pair,
    s_close_pair,
    lowrisk,
    safe_p,
    total_p,
    lat,
    lon,
    serialno,
    eventtime,
    posepredict,
    posepredictprob,
    maskPredict,
    maskPredictProb,
    EventProcessedUtcTime,
    EventEnqueuedUtcTime
INTO sqloutput
FROM sddinput

SELECT 
    sddinput.serialno,
    sqlreferenceinput.city,
    avg(CAST(sddinput.distance as bigint)) as Avgdistance,
    max(CAST(sddinput.distance as bigint)) as Maxdistance,
    min(CAST(sddinput.distance as bigint)) as Mindistance,
    COUNT(CAST(sddinput.highrisk as bigint)) as Countofhighrisk,
    AVG(CAST(sddinput.highrisk as bigint)) as Avghighrisk,
    COUNT(CAST(sddinput.safe_p as bigint)) as Countsafep,
    AVG(CAST(sddinput.safe_p as bigint)) as Avgsafep,
    COUNT(CAST(sddinput.total_p as bigint)) as Counttotalp,
    AVG(CAST(sddinput.total_p as bigint)) as Avgtotalp,
    min(CAST(sddinput.eventtime as datetime)) as StartTime,
    max(CAST(sddinput.eventtime as datetime)) as EndTime
INTO pbioutput
FROM sddinput inner join sqlreferenceinput
on sddinput.serialno = sqlreferenceinput.DeviceSerialNumber
Group by sddinput.serialno, sqlreferenceinput.city, TumblingWindow(minute, 1)

SELECT 
    serialno,
    avg(CAST(sddinput.distance as bigint)) as Avgdistance,
    max(CAST(sddinput.distance as bigint)) as Maxdistance,
    min(CAST(sddinput.distance as bigint)) as Mindistance,
    COUNT(CAST(sddinput.highrisk as bigint)) as Countofhighrisk,
    AVG(CAST(sddinput.highrisk as bigint)) as Avghighrisk,
    COUNT(CAST(sddinput.safe_p as bigint)) as Countsafep,
    AVG(CAST(sddinput.safe_p as bigint)) as Avgsafep,
    COUNT(CAST(sddinput.total_p as bigint)) as Counttotalp,
    AVG(CAST(sddinput.total_p as bigint)) as Avgtotalp,
    min(CAST(sddinput.eventtime as datetime)) as StartTime,
    max(CAST(sddinput.eventtime as datetime)) as EndTime
INTO aggrsqloutput
FROM sddinput
Group by serialno, TumblingWindow(minute, 1)
```

## Azure SQL Database

Now go to SQL and display the table data

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/sql1.jpg "SQL")

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/sql2.jpg "SQL")


## Azure Custom Vision model to detect Sitting or Standing position.

Create model using Azure cognitive service api - custom vision.

- Create a Cognitive service account. Select S1 plan
- Log into https://customvision.ai
- Create a new project called sittingposition with compact object detection model S1
- Create images on local computer. (i used camera app in windows 10 and took few pictures)
- upload images.
- Create 2 tags Sitting, Standing
- Tag the images with bounding box for sitting and standing.
- Click Train - select quick training
- Wait untill the model training is completed
- Go to prediction and click Export
- Select Tensorflow and download the zip file
- Zip file contains model.pb, labels.txt (tensor graph)
- Sample inferencing code is available in download
- to integrate with our sdd.py import tensorflow and other libs
- resize the frame to 320,320,3
- write code to inference and save the predicted class and probablity and send that to event hub
- Code below has to go inside run function. (check line 261)

```
import argparse
import tensorflow as tf
import numpy as np
import PIL.Image

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

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
```

Model predicted and it's output

## Model Output for Sitting/Standing

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/sql3.jpg "SQL")

## Power BI

## Report

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/powerbi2.jpg "Batch report")

## Realtime Report

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/powerbirealtime1.jpg "RealTime report")

## Power Apps

Here is sample power apps code.

- Log into Power Apps Designer
- Create a data source to Azure SQL Database
- Create a canvas app
- Connect to the SQL data base
- Create a formula to if high risk count is > 0 then change the color to Red other wise make it green

```
in the formula: If(Value(ThisItem.highrisk) > 0, Red, Green)
```

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/powerapps1.jpg "Power Apps")

Detailed

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/powerapps2.jpg "Power Apps")

## ToDo

More to come.

- work environment health work habits. Sitting vs standing ratio analysis for improve employee health
- Work desk lighting
- Tiredness detection
- Stress detection
- Alergy detection
- Sneeze and Cough detection

## Credit

- Yolo
- Nadav Loebl - nadavloebl@gmail.com (MS) for Social Distance calcualtion.

## Thanks to contributors

- Priya Aswani
- Balamurugan Balakreshnan
- Shruti Harish
- Mickey Patel
- Prateek Gandhi
- Samson leung
- Malek el Khazen
- Dee Kumar
- Abraham Pabbathi
- Steve Thompson
