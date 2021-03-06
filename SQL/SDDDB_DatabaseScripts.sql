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
