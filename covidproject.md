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
