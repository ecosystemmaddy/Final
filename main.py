# main.py
# GIS Project Starter - Boundary by State / City / County FIPS
# Downloads TIGER 2020 data, builds a boundary from user input,
# and clips counties + state-level primary/secondary roads to that boundary.

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
    # PRISECROADS handled per-state using STATEFP (01, 02, ..., 72)
}
# Output folders
PROJECT_FOLDER = "GIS_Project_Starter"
DOWNLOAD_FOLDER = os.path.join(PROJECT_FOLDER, "downloads")
CLIPPED_FOLDER = os.path.join(PROJECT_FOLDER, "clipped")

# Ensures directories exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPPED_FOLDER, exist_ok=True)


# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def download_zip(name: str, url: str) -> str:
    """Download a ZIP file to downloads/<name>.zip, unless it already exists."""
    local_path = os.path.join(DOWNLOAD_FOLDER, f"{name}.zip")

    # Avoids redownloading
    if os.path.exists(local_path):
        print(f"{name}: ZIP already exists, skipping download.")
        return local_path

    print(f"Downloading {name} from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status() # stops program if request fails

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"{name}: downloaded to {local_path}")
    return local_path


def unzip_zip(name: str, zip_path: str) -> str:
    """Unzip ZIP into downloads/<name>/ and return the first .shp path."""
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

    # Try full name
    mask_name = states_gdf["NAME"].str.lower() == text.lower()
    subset = states_gdf[mask_name]
    if not subset.empty:
        return str(subset.iloc[0]["STATEFP"]).zfill(2)

    # Try postal code
    mask_code = states_gdf["STUSPS"].str.upper() == text.upper()
    subset = states_gdf[mask_code]
    if not subset.empty:
        return str(subset.iloc[0]["STATEFP"]).zfill(2)

    # If neither works
    raise ValueError(
        f"Could not resolve state '{user_input}'. "
        "Use full name (e.g., Texas) or 2-letter code (e.g., TX)."
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
        # Select and dissolve state boundary
        boundary = states[states["STATEFP"] == state_fips]

        if boundary.empty:
            raise ValueError(f"No state found for '{boundary_value}'.")

        boundary = boundary.dissolve().reset_index(drop=True)
        print(f"Boundary: state '{boundary_value}' (STATEFP={state_fips}), features: {len(boundary)}")
        return boundary

    elif boundary_type == "fips":
        # Ensure valid county fips format
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

    # Match city by state and name
    mask_name = places["NAME"].str.lower() == city_name.strip().lower()
    mask_state = places["STATEFP"] == state_fips
    subset = places[mask_name & mask_state]

    if subset.empty:
        raise ValueError(
            f"No place named '{city_name}' found in state '{state_input}' "
            f"(STATEFP={state_fips})."
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
) -> None:
    """Clip a TIGER layer to a polygon boundary, then enforce single geometry type."""
    print(f"Clipping {layer_shp} ...")
    base = gpd.read_file(layer_shp)
    print(f"  {layer_shp}: {len(base)} features before clip")

    # CRS must match or clip will be wrong
    if base.crs is None or boundary_gdf.crs is None:
        raise ValueError("CRS missing on base or boundary")

    # Align CRS if needed
    if base.crs != boundary_gdf.crs:
        boundary = boundary_gdf.to_crs(base.crs)
    else:
        boundary = boundary_gdf

    # Actual clip
    clipped = gpd.clip(base, boundary)
    print(f"  {output_path}: {len(clipped)} features after clip (before geometry filter)")

    # Decide what geometry family to keep based on base layer
    geom_types = set(base.geom_type.unique())
    polygonish = {"Polygon", "MultiPolygon"}
    lineish = {"LineString", "MultiLineString"}
    pointish = {"Point", "MultiPoint"}

    if geom_types <= polygonish:
        allowed = polygonish
    elif geom_types <= lineish:
        allowed = lineish
    elif geom_types <= pointish:
        allowed = pointish
    else:
        allowed = geom_types  # fallback

    clipped = clipped[clipped.geometry.type.isin(allowed)].copy()
    print(f"  {output_path}: {len(clipped)} features after filtering to {allowed}")

    # Save output
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_path)
    print(f"Saved clipped file: {output_path}")


# ---------------------------------
# MAIN
# ---------------------------------

def main() -> None:
    print("=== GIS Project Starter (TIGER 2020, State / City / County FIPS) ===\n")

    # Choose method of defining boundary
    print("Choose boundary type:")
    print("  1) State  (by name or code, e.g., Texas or TX)")
    print("  2) City   (requires city name AND state)")
    print("  3) FIPS   (5-digit county FIPS, e.g., 48113)")
    choice = input("Enter 'state', 'city', or 'fips' (or 1/2/3): ").strip().lower()

    # Parse mode
    if choice in ("1", "state"):
        mode = "state"
        state_input = input("Enter state name or 2-letter code (e.g., Texas or TX): ").strip()
    elif choice in ("2", "city"):
        mode = "city"
        city_name = input("Enter city/place name (e.g., Dallas): ").strip()
        city_state = input("Enter state name or code for that city (e.g., Texas or TX): ").strip()
    elif choice in ("3", "fips"):
        mode = "fips"
        fips_value = input("Enter 5-digit county FIPS code (e.g., 48113): ").strip()
    else:
        raise ValueError("Invalid choice. Use 'state', 'city', or 'fips' (or 1/2/3).")

    # Step 1: Download + unzip national states, counties
    states_zip = download_zip("states", LAYER_URLS["states"])
    states_shp = unzip_zip("states", states_zip)

    counties_zip = download_zip("counties", LAYER_URLS["counties"])
    counties_shp = unzip_zip("counties", counties_zip)

    # Step 2: determine STATEFP for roads (PRISECROADS)
    states_gdf = gpd.read_file(states_shp)

    if mode == "state":
        state_fips = resolve_state_fips(states_gdf, state_input)
    elif mode == "city":
        state_fips = resolve_state_fips(states_gdf, city_state)
    else:  # mode == "fips"
        # First 2 digits of county GEOID = state FIPS
        state_fips = fips_value[:2]

    prisec_name = f"prisecroads_{state_fips}"
    prisec_url = f"{BASE_URL}/PRISECROADS/tl_2020_{state_fips}_prisecroads.zip"
    roads_zip = download_zip(prisec_name, prisec_url)
    roads_shp = unzip_zip(prisec_name, roads_zip)

    # Step 3: Build boundary
    if mode == "city":
        boundary = build_city_boundary(city_name, city_state, states_shp)
        boundary_type = "city"
        boundary_value = city_name
        boundary.to_file(os.path.join(CLIPPED_FOLDER, "city_boundary.shp"))

    elif mode == "state":
        boundary = build_boundary("state", state_input, states_shp, counties_shp)
        boundary_type = "state"
        boundary_value = state_input
    else:  # mode == "fips"
        boundary = build_boundary("fips", fips_value, states_shp, counties_shp)
        boundary_type = "fips"
        boundary_value = fips_value

    # Optional: debug boundary
    # boundary.to_file(os.path.join(CLIPPED_FOLDER, "boundary_debug.shp"))

    print(f"\nUsing boundary type: {boundary_type}, value: {boundary_value}")
    print(f"Using state FIPS {state_fips} for PRISECROADS\n")

    # Step 4: Clip roads + counties
    print("Clipping layers to boundary...\n")

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

