# GIS Project Starter – TIGER 2020 Boundary Clipping Tool

## 1. Purpose of the Program

This program automates downloading and processing TIGER/Line 2020 geospatial datasets from the U.S. Census Bureau. It allows the user to define a geographic boundary using one of three inputs:

- A U.S. **state** (full name or 2-letter postal code)
- A **city/place** within a given state
- A **county** using its 5-digit **FIPS** (GEOID) code

The script then:
- Downloads the required TIGER/Line datasets  
- Constructs the corresponding polygon boundary  
- Clips primary/secondary roads and counties to that boundary  
- Outputs shapefiles ready for GIS analysis

Its purpose is to demonstrate automated geospatial data acquisition, boundary construction, CRS alignment, and vector clipping using Python and GeoPandas.

---

## 2. Intended / Potential Users

This program is suited for:

- **GIS students** learning Python-based geoprocessing  
- **Instructors** evaluating student work with TIGER/Line and spatial clipping  
- **Analysts** who want reproducible, script-based boundary clipping  
- Anyone needing a simple tool to generate boundary-specific geospatial layers

Only basic GIS and Python familiarity is required.

---

## 3. Programming and Execution Environment

### 3.1 Programming Language
- Python **3.10+**

### 3.2 Required Packages
- `geopandas`
- `requests`
- `shapely` (GeoPandas dependency)
- `fiona` (GeoPandas dependency)
- `pyproj`
- `pandas`

### 3.3 Recommended Installation

**Using Conda (recommended):**
conda create -n gis_project python=3.11
conda activate gis_project
conda install -c conda-forge geopandas requests

powershell
Copy code

**Using pip (only if GDAL stack already configured):**
pip install geopandas requests

yaml
Copy code

### 3.4 Operating System
Compatible with **Windows**, **macOS**, and **Linux**.

### 3.5 Internet Requirement
Internet access is required on the **first run** to download TIGER/Line files.  
Downloaded files are cached and reused.

---

## 4. Running the Program

### 4.1 Project Structure (created automatically)
GIS_Project_Starter/
├── downloads/ # raw TIGER ZIPs and extracted shapefiles
└── clipped/ # final clipped outputs

css
Copy code

### 4.2 Running the Script
From the folder containing `main.py`:
python main.py

markdown
Copy code

You will be prompted to choose:
- `state`
- `city`
- `fips`

### Example Inputs

**State Mode**
Enter 'state', 'city', or 'fips': state
Enter state name or 2-letter code: Texas

markdown
Copy code

**City Mode**
Enter 'state', 'city', or 'fips': city
Enter city/place name: Dallas
Enter state name or code: Texas

markdown
Copy code

**County FIPS Mode**
Enter 'state', 'city', or 'fips': fips
Enter 5-digit county FIPS code: 48113

yaml
Copy code

---

## 5. Input and Output Files

### 5.1 Inputs
The program automatically downloads:

- `tl_2020_us_state.zip`
- `tl_2020_us_county.zip`
- `tl_2020_<STATEFP>_prisecroads.zip`
- `tl_2020_<STATEFP>_place.zip` *(city mode)*

No initial datasets are required.

### 5.2 Outputs
All results are written to:
GIS_Project_Starter/clipped/

yaml
Copy code

Outputs include:

- `roads_clipped.shp` — PRISECROADS clipped to boundary  
- `counties_clipped.shp` — counties clipped to boundary  
- `city_boundary.shp` — city boundary polygon (city mode only)

---

## 6. Data Sources and Reference Code

### 6.1 TIGER/Line Data Sources
Data is retrieved from:

https://www2.census.gov/geo/tiger/TIGER2020/

Directories used:
- `STATE/`
- `COUNTY/`
- `PRISECROADS/`
- `PLACE/`

### 6.2 Reference Code
All code was written specifically for this project.  
Concepts are based on standard usage patterns from:
- GeoPandas documentation  
- Python `requests` examples  
- TIGER/Line attribute documentation (STATEFP, GEOID, NAME)

---

## 7. Notes for Instructor / Evaluator

### 7.1 How to Verify
1. Install Python environment and dependencies.  
2. Run the script.  
3. Test all three boundary modes.  
4. Open the resulting shapefiles in QGIS or ArcGIS Pro.  
5. Confirm:
   - Boundary polygon is correct  
   - Roads and counties are clipped correctly  
   - City boundary exists in city mode  

### 7.2 Assumptions and Limitations
- City matching is exact (case-insensitive).  
- TIGER URLs must remain valid.  
- Internet needed for initial download.  
- CRS is assumed to be present in TIGER files (it is).  

### 7.3 Dataset Submission Guidance
The script downloads all required datasets automatically.  
If required by course policy, you may submit:
- Example clipped outputs  
- A ZIP of your `clipped/` directory  

---

## 8. File Overview

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

---
