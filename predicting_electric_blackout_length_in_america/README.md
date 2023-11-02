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
