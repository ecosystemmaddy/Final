# main.py
# GIS Project Starter - Boundary by State / City / County FIPS
# Downloads TIGER 2020 data, builds a boundary from user input,
# and clips roads + counties to that boundary.

import os
import zipfile
from pathlib import Path

import requests
import geopandas as gpd

# ---------------------------------
# CONFIGURATION
# ---------------------------------

BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020"

LAYER_URLS = {
    "states":   f"{BASE_URL}/STATE/tl_2020_us_state.zip",
    "counties": f"{BASE_URL}/COUNTY/tl_2020_us_county.zip",
    "roads":    f"{BASE_URL}/PRIMARYROADS/tl_2020_us_primaryroads.zip",
    # places is handled separately because it's per-state using STATEFP
}

PROJECT_FOLDER = "GIS_Project_Starter"
DOWNLOAD_FOLDER = os.path.join(PROJECT_FOLDER, "downloads")
CLIPPED_FOLDER = os.path.join(PROJECT_FOLDER, "clipped")

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPPED_FOLDER, exist_ok=True)


# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def download_zip(name: str, url: str) -> str:
    """
    Download a ZIP file to downloads/<name>.zip, unless it already exists.
    Returns the local ZIP path.
    """
    local_path = os.path.join(DOWNLOAD_FOLDER, f"{name}.zip")
    if os.path.exists(local_path):
        print(f"{name}: ZIP already exists, skipping download.")
        return local_path

    print(f"Downloading {name} from {url} ...")
    resp = requests.get(url, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to download {name} ({url}): {e}")

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"{name}: downloaded to {local_path}")
    return local_path


def unzip_zip(name: str, zip_path: str) -> str:
    """
    Unzip ZIP into downloads/<name>/ and return the path
    to the first .shp found (recursively).
    """
    extract_folder = os.path.join(DOWNLOAD_FOLDER, name)
    os.makedirs(extract_folder, exist_ok=True)

    print(f"Unzipping {zip_path} to {extract_folder} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_folder)
    print(f"{name}: unzip complete.")

    shp_path = find_shapefile(extract_folder)
    if not shp_path:
        raise FileNotFoundError(f"No .shp found in {extract_folder} for {name}")

    print(f"{name}: using shapefile {shp_path}")
    return shp_path


def find_shapefile(folder: str) -> str | None:
    """Return the first .shp file found in folder (recursive), or None."""
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".shp"):
                return os.path.join(root, f)
    return None


def resolve_state_fips(states_gdf: gpd.GeoDataFrame, user_input: str) -> str:
    """
    Given states GeoDataFrame and a user input like 'Texas' or 'TX',
    return the STATEFP code as a zero-padded string (e.g., '48').
    """
    text = user_input.strip()
    # Try match by NAME (full name)
    mask_name = states_gdf["NAME"].str.lower() == text.lower()
    subset = states_gdf[mask_name]
    if not subset.empty:
        fips = subset.iloc[0]["STATEFP"]
        return str(fips).zfill(2)

    # Try match by STUSPS (postal, e.g., 'TX')
    mask_code = states_gdf["STUSPS"].str.upper() == text.upper()
    subset = states_gdf[mask_code]
    if not subset.empty:
        fips = subset.iloc[0]["STATEFP"]
        return str(fips).zfill(2)

    raise ValueError(
        f"Could not resolve state '{user_input}'. "
        f"Use full name (e.g., Texas) or 2-letter code (e.g., TX)."
    )


def build_boundary(
    boundary_type: str,
    boundary_value: str,
    states_shp: str,
    counties_shp: str,
) -> gpd.GeoDataFrame:
    """
    Build a boundary GeoDataFrame for:
      - state: state name or postal (Texas, TX)
      - fips:  5-digit county FIPS (e.g., 48113)
    """
    boundary_type = boundary_type.lower().strip()
    boundary_value = boundary_value.strip()

    if boundary_type == "state":
        states = gpd.read_file(states_shp)
        state_fips = resolve_state_fips(states, boundary_value)
        boundary = states[states["STATEFP"] == state_fips]

        if boundary.empty:
            raise ValueError(f"No state found for '{boundary_value}'.")

        boundary = boundary.dissolve().reset_index(drop=True)
        print(f"Boundary: state '{boundary_value}' (STATEFP={state_fips}), features: {len(boundary)}")
        return boundary

    elif boundary_type == "fips":
        if len(boundary_value) != 5 or not boundary_value.isdigit():
            raise ValueError("County FIPS must be a 5-digit numeric code, e.g., 48113.")

        counties = gpd.read_file(counties_shp)
        subset = counties[counties["GEOID"] == boundary_value]

        if subset.empty:
            raise ValueError(
                f"No county found with FIPS '{boundary_value}'. "
                "Use the 5-digit GEOID value."
            )

        boundary = subset.dissolve().reset_index(drop=True)
        print(f"Boundary: county GEOID={boundary_value}, features: {len(boundary)}")
        return boundary

    else:
        raise ValueError("build_boundary only supports 'state' or 'fips' here.")


def build_city_boundary(
    city_name: str,
    state_input: str,
    states_shp: str,
) -> gpd.GeoDataFrame:
    """
    Build a city/place boundary:
      - city_name: e.g., 'Dallas'
      - state_input: state name or code, e.g., 'Texas' or 'TX'
    Uses state file to get STATEFP, then downloads the correct PLACE ZIP.
    """
    states = gpd.read_file(states_shp)
    state_fips = resolve_state_fips(states, state_input)

    # Construct PLACE URL for this state
    place_url = f"{BASE_URL}/PLACE/tl_2020_{state_fips}_place.zip"
    place_name = f"places_{state_fips}"

    # Download + unzip this state's PLACE file
    zip_path = download_zip(place_name, place_url)
    place_shp = unzip_zip(place_name, zip_path)

    places = gpd.read_file(place_shp)

    # Filter by city NAME and matching STATEFP
    mask_name = places["NAME"].str.lower() == city_name.strip().lower()
    mask_state = places["STATEFP"] == state_fips
    subset = places[mask_name & mask_state]

    if subset.empty:
        raise ValueError(
            f"No place named '{city_name}' found in state '{state_input}' "
            f"(STATEFP={state_fips}). Remember this uses Census 'place' names."
        )

    boundary = subset.dissolve().reset_index(drop=True)
    print(
        f"Boundary: city '{city_name}' in state '{state_input}' "
        f"(STATEFP={state_fips}), features: {len(boundary)}"
    )
    return boundary


def clip_layer_to_boundary(
    layer_shp: str,
    boundary_gdf: gpd.GeoDataFrame,
    output_path: str,
):
    """Clip any TIGER layer to a polygon boundary using geopandas.clip."""
    print(f"Clipping {layer_shp} ...")
    base = gpd.read_file(layer_shp)

    # Align CRS
    if base.crs != boundary_gdf.crs:
        boundary = boundary_gdf.to_crs(base.crs)
    else:
        boundary = boundary_gdf

    clipped = gpd.clip(base, boundary)

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_path)
    print(f"Saved clipped file: {output_path} (features: {len(clipped)})")


# ---------------------------------
# MAIN
# ---------------------------------

def main():
    print("=== GIS Project Starter (TIGER 2020, State / City / County FIPS) ===\n")

    print("Choose boundary type:")
    print("  1) State  (by name or code, e.g., Texas or TX)")
    print("  2) City   (requires city name AND state)")
    print("  3) FIPS   (5-digit county FIPS, e.g., 48113)")
    choice = input("Enter 'state', 'city', or 'fips' (or 1/2/3): ").strip().lower()

    if choice in ("1", "state"):
        boundary_type = "state"
        state_input = input("Enter state name or 2-letter code (e.g., Texas or TX): ").strip()
        city_mode = False
        fips_value = None
        city_name = None
        city_state = None
    elif choice in ("2", "city"):
        boundary_type = "city"
        city_name = input("Enter city/place name (e.g., Dallas): ").strip()
        city_state = input("Enter state name or code for that city (e.g., Texas or TX): ").strip()
        city_mode = True
        fips_value = None
        state_input = None
    elif choice in ("3", "fips"):
        boundary_type = "fips"
        fips_value = input("Enter 5-digit county FIPS code (e.g., 48113): ").strip()
        city_mode = False
        state_input = None
        city_name = None
        city_state = None
    else:
        raise ValueError("Invalid choice. Use 'state', 'city', or 'fips' (or 1/2/3).")

    # Step 1: Download + unzip the national layers we always need
    # States, counties, roads
    states_zip = download_zip("states", LAYER_URLS["states"])
    states_shp = unzip_zip("states", states_zip)

    counties_zip = download_zip("counties", LAYER_URLS["counties"])
    counties_shp = unzip_zip("counties", counties_zip)

    roads_zip = download_zip("roads", LAYER_URLS["roads"])
    roads_shp = unzip_zip("roads", roads_zip)

    # Step 2: Build boundary
    if city_mode:
        boundary = build_city_boundary(city_name, city_state, states_shp)
    else:
        boundary = build_boundary(boundary_type, state_input or fips_value, states_shp, counties_shp)

    # Step 3: Clip roads and counties to boundary
    print("\nClipping layers to boundary...\n")
    try:
        clip_layer_to_boundary(roads_shp, boundary, os.path.join(CLIPPED_FOLDER, "roads_clipped.shp"))
    except Exception as e:
        print(f"Error clipping roads: {e}")

    try:
        clip_layer_to_boundary(counties_shp, boundary, os.path.join(CLIPPED_FOLDER, "counties_clipped.shp"))
    except Exception as e:
        print(f"Error clipping counties: {e}")

    print("\nAll done.")
    print(f"Clipped outputs saved in: {os.path.abspath(CLIPPED_FOLDER)}")


if __name__ == "__main__":
    main()
