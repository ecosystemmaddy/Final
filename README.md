# GIS Project Starter – TIGER 2020 Boundary Clipping Tool

## Description

This program provides a Python-based workflow for automatically downloading, processing, and clipping TIGER/Line 2020 geopsatial datasets from the U.S. Census Bureau.  It allows the user to define a geographic boundary using one of three inputs:

- A U.S. **state** (full name or 2-letter postal code)
- A **city/place** within a given state (City and State)
- A **county** using its 5-digit **FIPS** (GEOID) code

The script finds the necessary TIGER/Line shapefiles and constructs the boundary polygon, aligns coordinate reference systems, clips both roads and counties to the selected area, and outputs ready-to-use shapefiles for GIS analysis. 

The tool is intended for **GIS students**, **data analysts**, and **beginners** learning spatial data processing with Python. No advanced GIS background is required, only basic GIS and Python familiarity.

This project demonstrates key workflows in Python-based GIS, such as automated data acquisition, vector boundary modeling, CRS management, spatial overlay analysis, and file handling/production.


## Getting Started

### Dependencies
Before running the program, make sure you have: 
- **Python 3.10 or newer** (GeoPandas installation is easiest with these versions)
- Internet access for the first dataset download. Downloaded files are cached and reused.
- The following Python packages:
  - `geopandas`
  - `requests`
  - `shapely` (GeoPandas dependency)
  - `fiona` (GeoPandas dependency)
  - `pyproj`
  - `pandas`

### Environment Setup

**Using Conda (recommended):**  
conda create -n gis_project python=3.11  
conda activate gis_project  
conda install -c conda-forge geopandas requests  

**Using pip (only if GDAL is already working):**  
pip install geopandas requests  

### Operating System  
The program is compatible with **Windows**, **macOS**, and **Linux**.  

***Note:**
macOS and Linux may require additional system dependencies. Using Conda is strongly recomended since it already has all necessary libraires installed automatically. Pip installation may fail unless GDAL is already configured.*

### Installing
No installation is required beyond downloading the files. 
1. Download or copy the project folder containing
  - `main.py`
2. Ensure the project directory has write permissions
3. No modifications to files or folders are necessary. All `downloads/` and `clipped/` directories are generated automatically.


## File Overview

### `main.py`
Contains the full logic:
- Download functions  
- ZIP extraction  
- Boundary creation (state, city, FIPS)  
- Layer clipping  
- Console interaction  

### Input Files
The program automatically downloads:
- `tl_2020_us_state.zip`
- `tl_2020_us_county.zip`
- `tl_2020_<STATEFP>_prisecroads.zip`
- `tl_2020_<STATEFP>_place.zip` *(city mode)*

No initial datasets are required.

### Output Files
After running, the program generates:

**GIS_Project_Starter/  
├── downloads/** *-  the raw TIGER ZIPs and extracted shapefiles*   
**└── clipped/** *-  the final clipped outputs*  

Clipped output shapefiles include: 
- `roads_clipped.shp`
- `counties_clipped.shp`
- `city_boundary.shp` *(city mode)*

All results are written to:
GIS_Project_Starter/clipped/


## Executing program
From the project directory, run: 

python main.py

The script will then prompt for a boundary selection: 
- `state`
- `city`
- `fips`

## Example Usage
1. **State Mode**
Enter 'state', 'city', or 'fips': state
Enter state name or 2-letter code: Texas

2. **City Mode**
Enter 'state', 'city', or 'fips': city
Enter city/place name: Dallas
Enter state name or code: Texas

3. **County FIPS Mode**
Enter 'state', 'city', or 'fips': fips
Enter 5-digit county FIPS code: 48113


## Help
Common issues include: 

### GeoPandas fails to import
Install using Conda:
conda install -c conda-forge geopandas

### City not found
Ensure spelling matches TIGER data naming.

### FIPS rejected
FIPS must be eaxctly **5 digits** (e.g., 48113)

### No roads in output
Some small cities have no primary/secondary roads in PRISECROADSD.  
There is no built-in command, but you can try re-running the script if the input was wrong.


## Data Sources and Reference Code

### TIGER/Line Data Sources
Data is retrieved from:

https://www2.census.gov/geo/tiger/TIGER2020/

Directories used:
- `STATE/`
- `COUNTY/`
- `PRISECROADS/`
- `PLACE/`

### Reference Code
All code was written specifically for this project.  
Concepts are based on standard usage patterns from:
- GeoPandas documentation  
- Python `requests` examples  
- TIGER/Line attribute documentation (STATEFP, GEOID, NAME)

## Notes for Instructor

### How to Verify
1. Install Python environment and dependencies.  
2. Run the script.  
3. Test all three boundary modes.  
4. Open the resulting shapefiles in QGIS or ArcGIS Pro.  
5. Confirm:
   - Boundary polygon is correct  
   - Roads and counties are clipped correctly  
   - City boundary exists in city mode  

### Assumptions and Limitations
- City matching is exact (case-insensitive).  
- TIGER URLs must remain valid.  
- Internet needed for initial download.  
- CRS is assumed to be present in TIGER files (it is).

### Final File Overview
### `main.py`
Contains the full logic:
- Download functions  
- ZIP extraction  
- Boundary creation (state, city, FIPS)  
- Layer clipping  
- Console interaction  

### Runtime-created folders:
- `GIS_Project_Starter/downloads/` — downloaded TIGER data  
- `GIS_Project_Starter/clipped/` — final analysis outputs 


## Authors
**Madelyn Schumuhl**  
(Project Contributor)  
mjs220001@utdallas.edu  

**Abigail McFarland**  
(Project Contributor)  
arm220022@utdallas.edu
