"""
- Read CSV datasets 
- Download country border shapes
- Merge nums into the shapes using country codes
- Draw interactive 3D topographic globe based on filters
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests


# Default elevation scale
ELEVATION_SCALE = 150000

# Initialize data
POP_CSV = Path("data/World Population 1960-2023 by Country.csv")
GDP_CSV = Path("data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2.csv")
CONTINENTS_CSV = Path("data/continents2.csv")

# Get country borders 
GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)
GEOJSON_CACHE = Path("data/ne_110m_admin_0_countries.geojson")


def require_file(path: Path) -> None:
    # Crash if no files
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file: {path}\n\n"
            "Make sure your folder looks like:\n"
            "  world-globe/\n"
            "    main.py\n"
            "    data/\n"
            f"      {POP_CSV.name}\n"
            f"      {GDP_CSV.name}\n"
            f"      {CONTINENTS_CSV.name}\n"
        )


def download_geojson() -> dict:
    GEOJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if GEOJSON_CACHE.exists():
        with open(GEOJSON_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    # download 
    print("Downloading country borders GeoJSON")
    r = requests.get(GEOJSON_URL, timeout=60)
    r.raise_for_status()
    geojson = r.json()

    # Save to cache
    with open(GEOJSON_CACHE, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"Saved GeoJSON to {GEOJSON_CACHE}")
    return geojson


def wide_to_long(df: pd.DataFrame, code_col: str, value_name: str) -> pd.DataFrame:
    """
      Country Code | 1960 | 1961 | ... | 2023 -->       iso3 | year | population
    """
    year_cols = [c for c in df.columns if c.isdigit()]
    out = df[[code_col] + year_cols].melt(
        id_vars=[code_col],
        var_name="year",
        value_name=value_name,
    )
    out["year"] = out["year"].astype(int)
    return out


def read_population() -> pd.DataFrame:
    require_file(POP_CSV)
    pop_wide = pd.read_csv(POP_CSV)

    #  "Country Code" + year cols
    pop_long = wide_to_long(pop_wide, code_col="Country Code", value_name="population")
    pop_long.rename(columns={"Country Code": "iso3"}, inplace=True)

    # force numeric
    pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")
    return pop_long


def read_gdp_pc() -> pd.DataFrame:
    require_file(GDP_CSV)

    # Read the 4 lines
    gdp_wide = pd.read_csv(GDP_CSV, skiprows=4)
    gdp_wide = gdp_wide.loc[:, ~gdp_wide.columns.str.contains(r"^Unnamed")]

    gdp_long = wide_to_long(gdp_wide, code_col="Country Code", value_name="gdp_pc")
    gdp_long.rename(columns={"Country Code": "iso3"}, inplace=True)
    gdp_long["gdp_pc"] = pd.to_numeric(gdp_long["gdp_pc"], errors="coerce")
    return gdp_long


def read_continents() -> pd.DataFrame:
    require_file(CONTINENTS_CSV)
    cont = pd.read_csv(CONTINENTS_CSV)

    cont = cont[["alpha-3", "region"]].rename(
        columns={"alpha-3": "iso3", "region": "continent"}
    )
    return cont


def make_color(norm_t: float) -> list:
    norm_t = max(0.0, min(1.0, float(norm_t)))

    r = int(40 + 200 * norm_t)
    g = int(90 + 60 * (1 - norm_t))
    b = int(220 - 150 * norm_t)
    a = 190
    return [r, g, b, a]


def attach_metric_to_geojson(geojson: dict, df_year: pd.DataFrame, metric: str) -> dict:
    # Merge nums to map
    # pick the col to visualize
    values = df_year[["iso3", metric]].copy()
    values.rename(columns={metric: "value"}, inplace=True)
    values["value"] = pd.to_numeric(values["value"], errors="coerce").fillna(0.0)

    # log scale 2 make range more balanced.
    values["log_value"] = values["value"].apply(lambda x: math.log10(x + 1) if x > 0 else 0.0)

    # normalization range for colors 
    nz = values[values["log_value"] > 0]["log_value"]
    vmin = float(nz.min()) if len(nz) else 0.0
    vmax = float(nz.max()) if len(nz) else 1.0

    # elevation scaling (this is what makes "height")
    elevation_scale = ELEVATION_SCALE
    values["elevation"] = values["log_value"] * elevation_scale

    # Normalize 0..1 for color
    def normalize(v):
        if vmax <= vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    values["fill_color"] = values["log_value"].apply(lambda v: make_color(normalize(v)))

    # lookup dict
    lookup = values.set_index("iso3")[["value", "elevation", "fill_color"]].to_dict("index")

    # Attach to each GeoJSON feature
    for feature in geojson["features"]:
        iso3 = feature["properties"].get("ADM0_A3")  # key used by Natural Earth
        data = lookup.get(iso3)

        if not data:
            # Missing data -> low gray
            feature["properties"]["value"] = 0
            feature["properties"]["elevation"] = 0
            feature["properties"]["fill_color"] = [120, 120, 120, 70]
        else:
            feature["properties"]["value"] = float(data["value"])
            feature["properties"]["elevation"] = float(data["elevation"])
            feature["properties"]["fill_color"] = data["fill_color"]

    return geojson


def main():
    # parser 1st then add arguments
    parser = argparse.ArgumentParser(description="3D globe extruded by population or GDP per capita.")

    # add elev flag so u can crank mountains up/down without touching code
    parser.add_argument("--elev", type=float, default=250000, help="Elevation scale (bigger = taller)")

    parser.add_argument("--year", type=int, default=2023, help="Year to visualize (e.g. 2023)")
    parser.add_argument(
        "--metric",
        choices=["population", "gdp_pc"],
        default="population",
        help="Which metric to extrude: population or GDP per capita",
    )
    parser.add_argument(
        "--continent",
        default="All",
        help="Filter by continent (e.g. Europe, Asia, Africa, Americas, Oceania) or All",
    )
    parser.add_argument("--out", default="globe.html", help="Output HTML filename")

    args = parser.parse_args()

    # apply the CLI elev into the global scaling variable
    global ELEVATION_SCALE
    ELEVATION_SCALE = args.elev

    # Load CSVs
    pop = read_population()
    gdp = read_gdp_pc()
    cont = read_continents()

    # Merge them into a table for easy filtering
    df = pop.merge(gdp, on=["iso3", "year"], how="outer")
    df = df.merge(cont, on="iso3", how="left")

    # Filter by continent if asked
    if args.continent != "All":
        df = df[df["continent"].fillna("") == args.continent]

    # Filter by year
    df_year = df[df["year"] == args.year].copy()

    # Download borders
    geojson = download_geojson()

    # Attach value/elevation/color 2 GeoJSON 
    geojson = attach_metric_to_geojson(geojson, df_year, metric=args.metric)

    # 3D layer
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,      # hover
        stroked=True,       # borders
        filled=True,        # fill 
        extruded=True,      # raised
        wireframe=True,
        get_fill_color="properties.fill_color",
        get_elevation="properties.elevation",
        get_line_color=[30, 30, 30, 160],
        line_width_min_pixels=0.3,
        auto_highlight=True
    )

    # Hover results
    pretty_metric = "Population" if args.metric == "population" else "GDP per capita (US$)"
    tooltip = {
        "html": "<b>{NAME}</b><br/>" + f"{pretty_metric}: " + "{properties.value}",
        "style": {"backgroundColor": "rgba(20,20,20,0.85)", "color": "white"},
    }

    # Camera view (more pitch + zoom so height is visible)
    view_state = pdk.ViewState(latitude=15, longitude=0, zoom=1.2, pitch=65)

    # Create the deck.gl scene w polygons
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        views=[pdk.View(type="MapView", controller=True)],
        map_style=None,
    )

    # Make HTML (offline=True so no CDN weirdness)
    deck.to_html(args.out, open_browser=False, offline=True)

    print("\nDone!")
    print(f"  Wrote: {args.out}")
    print("  Open it with:")
    print(f"    open {args.out}\n")


if __name__ == "__main__":
    main()
