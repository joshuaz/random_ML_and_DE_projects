# Overview
This Exercise seeks to predict the duration of outage events (in minutes)

## Data sources
[data source background](https://www.sciencedirect.com/science/article/pii/S2352340918307182)<br>
[data download](https://engineering.purdue.edu/LASCI/research-data/outages)

## Data Dictionary
### General Information
#### Time of the outage event
| Variable | Description |
| ----------- | ----------- |
| YEAR | Indicates the year when the outage event occurred |
| MONTH | Indicates the month when the outage event occurred |

#### Geographic Areas
| Variable | Description |
| ----------- | ----------- |
| U.S._STATE | Represents all the states in the continental U.S. |
| POSTAL.CODE | Represents the postal code of the U.S. states |
| NERC.REGION | The North American Electric Reliability Corporation (NERC) regions involved in the outage event |


### REGIONAL CLIMATE INFORMATION
#### U.S. Climate regions
| Variable | Description |
| ----------- | ----------- |
| CLIMATE.REGION | U.S. Climate regions as specified by National Centers for Environmental Information (nine climatically consistent regions in continental U.S.A.) |

#### El Niño/La Niña
| Variable | Description |
| ----------- | ----------- |
| ANOMALY.LEVEL | This represents the oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season. It is estimated as a 3-month running mean of ERSST.v4 SST anomalies in the Niño 3.4 region (5°N to 5°S, 120–170°W) [6] |
| CLIMATE.CATEGORY | This represents the climate episodes corresponding to the years. The categories—“Warm”, “Cold” or “Normal” episodes of the climate are based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI) |

### OUTAGE EVENTS INFORMATION
#### Event start and end information
| Variable | Description |
| ----------- | ----------- |
| OUTAGE.START.DATE | This variable indicates the day of the year when the outage event started (as reported by the corresponding Utility in the region) |
| OUTAGE.START.TIME | This variable indicates the time of the day when the outage event started (as reported by the corresponding Utility in the region) |
| OUTAGE.RESTORATION.DATE | This variable indicates the day of the year when power was restored to all the customers (as reported by the corresponding Utility in the region) |
| OUTAGE.RESTORATION.TIME | This variable indicates the time of the day when power was restored to all the customers (as reported by the corresponding Utility in the region) |

#### Cause of the event
| Variable | Description |
| ----------- | ----------- |
| CAUSE.CATEGORY | Categories of all the events causing the major power outages |
| CAUSE.CATEGORY.DETAIL | Detailed description of the event categories causing the major power outages |
| HURRICANE.NAMES | If the outage is due to a hurricane, then the hurricane name is given by this variable |

#### Extent of outages
| Variable | Description |
| ----------- | ----------- |
| OUTAGE.DURATION | Duration of outage events (in minutes) |
| DEMAND.LOSS.MW | Amount of peak demand lost during an outage event (in Megawatt) [but in many cases, total demand is reported] |
| CUSTOMERS.AFFECTED | Number of customers affected by the power outage event |

### REGIONAL ELECTRICITY CONSUMPTION INFORMATION
#### Electricity price
| Variable | Description |
| ----------- | ----------- |
| RES.PRICE | Monthly electricity price in the residential sector (cents/kilowatt-hour) |
| COM.PRICE | Monthly electricity price in the commercial sector (cents/kilowatt-hour) |
| IND.PRICE | Monthly electricity price in the industrial sector (cents/kilowatt-hour) |
| TOTAL.PRICE | Average monthly electricity price in the U.S. state (cents/kilowatt-hour) |

#### Electricity consumption
| Variable | Description |
| ----------- | ----------- |
| RES.SALES | Electricity consumption in the residential sector (megawatt-hour) |
| COM.SALES | Electricity consumption in the commercial sector (megawatt-hour) |
| IND.SALES | Electricity consumption in the industrial sector (megawatt-hour) |
| TOTAL.SALES | Total electricity consumption in the U.S. state (megawatt-hour) |
| RES.PERCEN | Percentage of residential electricity consumption compared to the total electricity consumption in the state (in %) |
| COM.PERCEN | Percentage of commercial electricity consumption compared to the total electricity consumption in the state (in %) |
| IND.PERCEN | Percentage of industrial electricity consumption compared to the total electricity consumption in the state (in %) |

#### Customers Served
| Variable | Description |
| ----------- | ----------- |
| RES.CUSTOMERS | Annual number of customers served in the residential electricity sector of the U.S. state |
| COM.CUSTOMERS | Annual number of customers served in the commercial electricity sector of the U.S. state |
| IND.CUSTOMERS | Annual number of customers served in the industrial electricity sector of the U.S. state |
| TOTAL.CUSTOMERS | Annual number of total customers served in the U.S. state |
| RES.CUST.PCT | Percent of residential customers served in the U.S. state (in %) |
| COM.CUST.PCT | Percent of commercial customers served in the U.S. state (in %) |
| IND.CUST.PCT | Percent of industrial customers served in the U.S. state (in %) |


### REGIONAL ECONOMIC CHARACTERISTICS
#### Economic outputs
| Variable | Description |
| ----------- | ----------- |
| PC.REALGSP.STATE | Per capita real gross state product (GSP) in the U.S. state (measured in 2009 chained U.S. dollars) |
| PC.REALGSP.USA | Per capita real GSP in the U.S. (measured in 2009 chained U.S. dollars) |
| PC.REALGSP.REL | Relative per capita real GSP as compared to the total per capita real GDP of the U.S. (expressed as fraction of per capita State real GDP & per capita US real GDP) |
| PC.REALGSP.CHANGE | Percentage change of per capita real GSP from the previous year (in %) |
| UTIL.REALGSP | Real GSP contributed by Utility industry (measured in 2009 chained U.S. dollars) |
| TOTAL.REALGSP | Real GSP contributed by all industries (total) (measured in 2009 chained U.S. dollars) |
| UTIL.CONTRI | Utility industry׳s contribution to the total GSP in the State (expressed as percent of the total real GDP that is contributed by the Utility industry) (in %) |
| PI.UTIL.OFUSA | State utility sector׳s income (earnings) as a percentage of the total earnings of the U.S. utility sector׳s income (in %) | 

### REGIONAL LAND-USE CHARACTERICS
#### Population
| Variable | Description |
| ----------- | ----------- |
| POPULATION | Population in the U.S. state in a year |
| POPPCT_URBAN | Percentage of the total population of the U.S. state represented by the urban population (in %) |
| POPPCT_UC | Percentage of the total population of the U.S. state represented by the population of the urban clusters (in %) |
| POPDEN_URBAN | Population density of the urban areas (persons per square mile) |
| POPDEN_UC | Population density of the urban clusters (persons per square mile) |
| POPDEN_RURAL | Population density of the rural areas (persons per square mile) |

#### Land Area
| Variable | Description |
| ----------- | ----------- |
| AREAPCT_URBAN | Percentage of the land area of the U.S. state represented by the land area of the urban areas (in %) |
| AREAPCT_UC | Percentage of the land area of the U.S. state represented by the land area of the urban clusters (in %) |
| PCT_LAND | Percentage of land area in the U.S. state as compared to the overall land area in the continental U.S. (in %) |
| PCT_WATER_TOT | Percentage of water area in the U.S. state as compared to the overall water area in the continental U.S. (in %) |
| PCT_WATER_INLAND | Percentage of inland water area in the U.S. state as compared to the overall inland water area in the continental U.S. (in %) |


