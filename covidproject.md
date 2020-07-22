# Employee Well Being - Covid workplace safety

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

## Architecture

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/hack2020.jpg "Architecture")

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

## Model Building

We are using python to build the model inferencing code. We are also leveraging yolo v3 tiny version.

wget https://pjreddie.com/media/files/yolov3-tiny.weights

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
    EventProcessedUtcTime datetime,
    EventEnqueuedUtcTime datetime
)
```

Stream Analytics

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
    EventProcessedUtcTime,
    EventEnqueuedUtcTime
INTO sqloutput
FROM sddinput
```

Now go to SQL and display the table data

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/sql1.jpg "SQL")

![alt text](https://github.com/balakreshnan/sdd/blob/master/images/sql2.jpg "SQL")