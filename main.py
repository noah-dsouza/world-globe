import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import pycountry


# Files
POP_CSV = Path("data/World Population 1960-2023 by Country.csv")
GDP_CSV = Path("data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2.csv")
CONTINENTS_CSV = Path("data/continents2.csv")

# Language list
COUNTRY_LANG_CSV = Path("data/countries-languages-spoken.csv")

# Borders
GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)
GEOJSON_CACHE = Path("data/ne_110m_admin_0_countries.geojson")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file: {path}\n\n"
            "Expected structure:\n"
            "  world-globe/\n"
            "    main.py\n"
            "    data/\n"
            f"      {POP_CSV.name}\n"
            f"      {GDP_CSV.name}\n"
            f"      {CONTINENTS_CSV.name}\n"
            f"      {COUNTRY_LANG_CSV.name}\n"
        )


def download_geojson() -> dict:
    GEOJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if GEOJSON_CACHE.exists():
        return json.loads(GEOJSON_CACHE.read_text(encoding="utf-8"))

    print("Downloading country borders GeoJSON…")
    r = requests.get(GEOJSON_URL, timeout=60)
    r.raise_for_status()
    geojson = r.json()

    GEOJSON_CACHE.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Saved GeoJSON to {GEOJSON_CACHE}")
    return geojson


def to_iso3(country_name: str) -> str | None:
    if not isinstance(country_name, str):
        return None
    name = country_name.strip()

    patches = {
        "United States of America": "United States",
        "United States": "United States",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Russia": "Russian Federation",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Bolivia": "Bolivia, Plurinational State of",
        "Iran": "Iran, Islamic Republic of",
        "Syria": "Syrian Arab Republic",
        "Tanzania": "Tanzania, United Republic of",
        "South Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Republic of",
        "Vietnam": "Viet Nam",
        "Laos": "Lao People's Democratic Republic",
        "Moldova": "Moldova, Republic of",
        "Czech Republic": "Czechia",
        "Ivory Coast": "Côte d'Ivoire",
        "Cape Verde": "Cabo Verde",
        "Palestine": "Palestine, State of",
        "Republic of the Congo": "Congo",
        "Democratic Republic of the Congo": "Congo, The Democratic Republic of the",
        "Eswatini": "Eswatini",
    }

    name = patches.get(name, name)

    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except Exception:
        return None


# CSV helpers
def wide_to_long(df: pd.DataFrame, code_col: str, value_name: str) -> pd.DataFrame:
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
    pop_long = wide_to_long(pop_wide, code_col="Country Code", value_name="population")
    pop_long.rename(columns={"Country Code": "iso3"}, inplace=True)
    pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")
    pop_long["iso3"] = pop_long["iso3"].astype(str).str.upper()
    return pop_long


def read_gdp_pc() -> pd.DataFrame:
    require_file(GDP_CSV)
    gdp_wide = pd.read_csv(GDP_CSV, skiprows=4)
    gdp_wide = gdp_wide.loc[:, ~gdp_wide.columns.str.contains(r"^Unnamed")]

    gdp_long = wide_to_long(gdp_wide, code_col="Country Code", value_name="gdp_pc")
    gdp_long.rename(columns={"Country Code": "iso3"}, inplace=True)
    gdp_long["gdp_pc"] = pd.to_numeric(gdp_long["gdp_pc"], errors="coerce")
    gdp_long["iso3"] = gdp_long["iso3"].astype(str).str.upper()
    return gdp_long


def read_continents() -> pd.DataFrame:
    require_file(CONTINENTS_CSV)
    cont = pd.read_csv(CONTINENTS_CSV)
    cont = cont[["alpha-3", "region"]].rename(columns={"alpha-3": "iso3", "region": "continent"})
    cont["iso3"] = cont["iso3"].astype(str).str.upper()
    return cont


def parse_languages_cell(cell: str) -> list[str]:
    """
    This dataset's language column is a big string like:
      "English, Spanish"
      "Arabic; French"
      "Portuguese / Spanish"
    We'll split on common delimiters and normalize.
    """
    if not isinstance(cell, str):
        return []
    s = cell.strip()
    if not s:
        return []

    # Normalize separators
    for sep in [";", "/", "|", " and "]:
        s = s.replace(sep, ",")

    parts = [p.strip() for p in s.split(",")]
    # Remove junk / normalize capitalization
    clean = []
    for p in parts:
        if not p:
            continue
        # remove extra annotations 
        p = p.split("(")[0].strip()
        if not p:
            continue
        clean.append(p)

    seen = set()
    out = []
    for lang in clean:
        key = lang.lower()
        if key not in seen:
            seen.add(key)
            out.append(lang)
    return out


def read_country_languages() -> tuple[dict[str, list[str]], list[str]]:
    require_file(COUNTRY_LANG_CSV)
    df = pd.read_csv(COUNTRY_LANG_CSV)

    # Find likely cols
    cols_lower = {c.lower(): c for c in df.columns}

    # country col
    country_col = None
    for key in ["country", "name", "country name"]:
        if key in cols_lower:
            country_col = cols_lower[key]
            break
    if country_col is None:
        country_col = df.columns[0]

    # languages col
    lang_col = None
    for key in ["languages spoken", "languages", "language", "spoken languages"]:
        if key in cols_lower:
            lang_col = cols_lower[key]
            break
    if lang_col is None:
        lang_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    iso3_to_langs: dict[str, list[str]] = {}
    all_langs_set = set()

    bad_rows = 0
    for _, r in df.iterrows():
        country = r.get(country_col, "")
        langs_raw = r.get(lang_col, "")

        iso3 = to_iso3(str(country))
        if not iso3:
            bad_rows += 1
            continue

        langs = parse_languages_cell(str(langs_raw))
        if not langs:
            continue

        iso3_to_langs[iso3] = langs
        for l in langs:
            all_langs_set.add(l.strip())

    all_languages = sorted(all_langs_set, key=lambda x: x.lower())

    if bad_rows:
        print(f"Could not ISO-map {bad_rows} rows from countries-languages-spoken.csv (name mismatches).")

    return iso3_to_langs, all_languages


def make_years(pop: pd.DataFrame) -> list[int]:
    return sorted(pop["year"].dropna().unique().astype(int).tolist())


def build_iso3_series(df_long: pd.DataFrame, years: list[int], value_col: str) -> dict[str, list[float]]:
    year_index = {y: i for i, y in enumerate(years)}
    out: dict[str, list[float]] = {}

    for iso3, sub in df_long.groupby("iso3"):
        arr = [0.0] * len(years)
        for _, row in sub.iterrows():
            y = int(row["year"])
            i = year_index.get(y)
            if i is not None:
                v = row[value_col]
                if pd.isna(v):
                    v = 0.0
                arr[i] = float(v)
        out[str(iso3).upper()] = arr

    return out


def attach_all_properties(
    geojson: dict,
    years: list[int],
    pop_series: dict[str, list[float]],
    gdp_series: dict[str, list[float]],
    continent_map: dict[str, str],
    iso3_to_langs: dict[str, list[str]],
) -> dict:
    for f in geojson["features"]:
        props = f.get("properties", {})
        iso3 = str(props.get("ADM0_A3") or "").upper()

        props["pop"] = pop_series.get(iso3, [0.0] * len(years))
        props["gdp"] = gdp_series.get(iso3, [0.0] * len(years))
        props["continent"] = continent_map.get(iso3, "Unknown")
        props["langs"] = iso3_to_langs.get(iso3, [])

        f["properties"] = props

    return geojson


def write_html(out_path: Path, geojson: dict, years: list[int], all_languages: list[str]) -> None:
    geojson_str = json.dumps(geojson)
    years_js = json.dumps(years)
    langs_js = json.dumps(all_languages)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>World Globe — Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: rgba(15, 18, 24, 0.78);
      --panel2: rgba(15, 18, 24, 0.55);
      --stroke: rgba(255,255,255,0.10);
      --stroke2: rgba(255,255,255,0.06);
      --text: #e6edf3;
      --muted: rgba(230,237,243,0.72);
      --muted2: rgba(230,237,243,0.55);
      --shadow: 0 18px 55px rgba(0,0,0,0.55);
      --radius: 18px;
      --radius2: 14px;
    }}

    html, body {{
      margin: 0; padding: 0; height: 100%;
      background: var(--bg);
      overflow: hidden;
      font-family: ui-sans-serif, system-ui, -apple-system;
      color: var(--text);
    }}

    #globe {{
      width: 100vw;
      height: 100vh;
    }}

    .brandChip {{
      position: fixed;
      left: 16px; top: 16px;
      z-index: 60;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 999px;
      background: var(--panel2);
      border: 1px solid var(--stroke2);
      backdrop-filter: blur(10px);
      pointer-events: none;
    }}
    .dot {{
      width: 10px; height: 10px;
      border-radius: 999px;
      background: linear-gradient(135deg, #63b3ff, #ff4f7a);
      box-shadow: 0 0 0 4px rgba(255,255,255,0.05);
    }}
    .brandChip .t {{
      font-weight: 900;
      font-size: 13px;
      letter-spacing: 0.2px;
    }}
    .brandChip .s {{
      color: var(--muted);
      font-size: 12px;
      margin-left: 10px;
      padding-left: 10px;
      border-left: 1px solid var(--stroke2);
      white-space: nowrap;
    }}

    .dock {{
      position: fixed;
      left: 16px;
      right: 16px;
      bottom: 16px;
      z-index: 70;
      background: var(--panel);
      border: 1px solid var(--stroke);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      padding: 12px 14px;
      pointer-events: auto;
    }}

    .dockTop {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}

    .pills {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--stroke2);
      color: var(--muted);
      white-space: nowrap;
    }}

    .pill strong {{
      color: var(--text);
      font-weight: 900;
    }}

    .dockBody {{
      display: grid;
      grid-template-columns: repeat(7, minmax(140px, 1fr));
      gap: 10px 12px;
      align-items: end;
    }}

    .control {{
      display: grid;
      gap: 6px;
    }}

    .control label {{
      font-size: 12px;
      color: var(--muted);
    }}

    select {{
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border: 1px solid var(--stroke2);
      border-radius: var(--radius2);
      padding: 10px 10px;
      outline: none;
    }}

    input[type="range"] {{
      accent-color: #63b3ff;
      width: 100%;
    }}

    button {{
      background: rgba(255,255,255,0.08);
      color: var(--text);
      border: 1px solid var(--stroke2);
      border-radius: var(--radius2);
      padding: 10px 12px;
      font-weight: 900;
      cursor: pointer;
      width: 100%;
    }}
    button:hover {{
      background: rgba(255,255,255,0.12);
    }}

    @media (max-width: 1200px) {{
      .dockBody {{
        grid-template-columns: repeat(3, minmax(140px, 1fr));
      }}
      .brandChip .s {{ display:none; }}
    }}
    @media (max-width: 620px) {{
      .dockBody {{
        grid-template-columns: repeat(2, minmax(140px, 1fr));
      }}
    }}

    #err {{
      position: fixed; right: 16px; top: 16px; z-index: 9999;
      background: rgba(255, 50, 50, 0.12); color: #ffd6d6;
      border: 1px solid rgba(255, 120, 120, 0.35);
      padding: 12px 14px; border-radius: 12px; max-width: 760px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas;
      display: none; white-space: pre-wrap;
      backdrop-filter: blur(10px);
    }}
  </style>

  <script type="module">
    const GEOJSON = {geojson_str};
    const YEARS = {years_js};
    const ALL_LANGS = {langs_js};

    const errBox = document.createElement('div');
    errBox.id = 'err';
    document.addEventListener('DOMContentLoaded', () => document.body.appendChild(errBox));
    function showErr(msg) {{
      const box = document.getElementById('err');
      if (!box) return;
      box.style.display = 'block';
      box.textContent = msg;
      console.error(msg);
    }}
    window.addEventListener('error', (e) => {{
      showErr("JS Error:\\n" + (e.message || e.error || "Unknown error"));
    }});
    window.addEventListener('unhandledrejection', (e) => {{
      showErr("Unhandled Promise Rejection:\\n" + (e.reason || "Unknown rejection"));
    }});

    const {{ default: Globe }} = await import("https://esm.sh/globe.gl@2.40.0?bundle");

    const state = {{
      metric: "population",
      yearIndex: YEARS.length - 1,
      continent: "All",
      language: "All",     // NEW: language filter
      maxalt: 0.12,
      power: 0.35
    }};

    function clamp(x, a, b) {{ return Math.max(a, Math.min(b, x)); }}

    function computeHeights(values, power) {{
      const v = values.map(x => Math.max(0, Number(x) || 0));
      const logv = v.map(x => Math.log10(x + 1));
      const nz = logv.filter(x => x > 0).sort((a,b)=>a-b);
      if (!nz.length) return logv.map(()=>0);

      const p = (arr, q) => {{
        const idx = (arr.length - 1) * q;
        const lo = Math.floor(idx);
        const hi = Math.ceil(idx);
        const t = idx - lo;
        return (1 - t) * arr[lo] + t * arr[hi];
      }};

      const lo = p(nz, 0.05);
      const hi = p(nz, 0.98);
      const denom = (hi - lo) || 1e-9;

      return logv.map(x => {{
        x = clamp(x, lo, hi);
        x = (x - lo) / denom;
        x = Math.pow(clamp(x, 0, 1), power);
        return x;
      }});
    }}

    function colorFromT(t) {{
      t = clamp(t, 0, 1);
      const r = Math.round(60 + 180 * t);
      const g = Math.round(160 - 110 * t);
      const b = Math.round(230 - 150 * t);
      const a = 220;
      return [r,g,b,a];
    }}

    function matchesContinent(feature) {{
      if (state.continent === "All") return true;
      const c = (feature.properties.continent || "Unknown");
      return c === state.continent;
    }}

    function matchesLanguage(feature) {{
      if (state.language === "All") return true;
      const langs = feature.properties.langs || [];
      // case-insensitive match
      return langs.some(l => (String(l).toLowerCase() === String(state.language).toLowerCase()));
    }}

    function rawValueFor(feature) {{
      const p = feature.properties;
      if (state.metric === "population") return (p.pop?.[state.yearIndex] ?? 0);
      if (state.metric === "gdp_pc") return (p.gdp?.[state.yearIndex] ?? 0);
      return 0;
    }}

    function metricLabel() {{
      if (state.metric === "population") return "Population";
      if (state.metric === "gdp_pc") return "GDP per capita (US$)";
      return "Value";
    }}

    const el = document.getElementById("globe");

    const globe = new Globe(el)
      .backgroundColor("#0b0f14")
      .globeImageUrl("https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-dark.jpg")
      .showAtmosphere(false)
      .polygonsData(GEOJSON.features)
      .polygonCapColor(d => {{
        const c = d.properties._col || [120,120,120,120];
        return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{c[3]/255}})`;
      }})
      .polygonSideColor(() => "rgba(0,0,0,0.0)")
      .polygonStrokeColor(() => "rgba(0,0,0,0.25)")
      .polygonAltitude(d => (d.properties._alt || 0) + 0.002)
      .polygonLabel(d => {{
        const name = d.properties.NAME || "Unknown";
        const iso3 = d.properties.ADM0_A3 || "";
        const v = d.properties._val || 0;
        const m = metricLabel();
        const langNote = (state.language !== "All") ? `<br/>Filter: <b>${{state.language}}</b>` : "";
        return `<b>${{name}} (${{iso3}})</b><br/>${{m}}: ${{Number(v).toLocaleString()}}${{langNote}}`;
      }});

    const c = globe.controls();
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.rotateSpeed = 0.55;
    c.zoomSpeed = 0.9;
    c.panSpeed = 0.8;
    c.minDistance = 180;
    c.maxDistance = 900;

    globe.pointOfView({{ lat: 15, lng: 0, altitude: 2.2 }});

    try {{
      const mat = globe.globeMaterial();
      mat.shininess = 0;
      mat.specular = {{ r: 0, g: 0, b: 0 }};
    }} catch (e) {{}}

    function renderHeader() {{
      const y = YEARS[state.yearIndex];
      const m = metricLabel();
      document.getElementById("pillMetric").innerHTML = `Metric <strong>${{m}}</strong>`;
      document.getElementById("pillYear").innerHTML = `Year <strong>${{y}}</strong>`;
      document.getElementById("pillCont").innerHTML = `Cont <strong>${{state.continent}}</strong>`;
      document.getElementById("pillLang").innerHTML = `Lang <strong>${{state.language}}</strong>`;
      document.getElementById("pillH").innerHTML = `Height <strong>${{state.maxalt.toFixed(2)}}</strong>`;
      document.getElementById("pillP").innerHTML = `Contrast <strong>${{state.power.toFixed(2)}}</strong>`;
    }}

    function updateComputedProps() {{
      const feats = GEOJSON.features;

      // active = continent match AND language match
      const active = feats.filter(f => matchesContinent(f) && matchesLanguage(f));

      // heights/colors computed ONLY on active set (so scaling looks good)
      const rawVals = active.map(rawValueFor);
      const tVals = computeHeights(rawVals, state.power);

      // Set active props
      active.forEach((f, i) => {{
        const t = tVals[i];
        f.properties._t = t;
        f.properties._alt = t * state.maxalt;
        f.properties._col = colorFromT(t);
        f.properties._val = rawValueFor(f);
      }});

      // Everything else dim + flat
      feats.forEach((f) => {{
        if (!(matchesContinent(f) && matchesLanguage(f))) {{
          f.properties._t = 0;
          f.properties._alt = 0;
          f.properties._col = [60, 60, 60, 60];
          f.properties._val = 0;
        }}
      }});

      globe.polygonsData(feats);
      renderHeader();
    }}

    function buildUI() {{
      const yearSlider = document.getElementById("year");
      const metricSel = document.getElementById("metric");
      const contSel = document.getElementById("continent");
      const langSel = document.getElementById("language");
      const heightSel = document.getElementById("height");
      const powerSel = document.getElementById("power");

      yearSlider.min = 0;
      yearSlider.max = YEARS.length - 1;
      yearSlider.value = state.yearIndex;

      heightSel.value = state.maxalt;
      powerSel.value = state.power;

      // Build language dropdown
      // Always include "All"
      langSel.innerHTML = "";
      const optAll = document.createElement("option");
      optAll.value = "All";
      optAll.textContent = "All";
      langSel.appendChild(optAll);

      ALL_LANGS.forEach(l => {{
        const opt = document.createElement("option");
        opt.value = l;
        opt.textContent = l;
        langSel.appendChild(opt);
      }});
      langSel.value = state.language;

      yearSlider.addEventListener("input", (e) => {{
        state.yearIndex = Number(e.target.value);
        updateComputedProps();
      }});

      metricSel.addEventListener("change", (e) => {{
        state.metric = e.target.value;
        updateComputedProps();
      }});

      contSel.addEventListener("change", (e) => {{
        state.continent = e.target.value;
        updateComputedProps();
      }});

      langSel.addEventListener("change", (e) => {{
        state.language = e.target.value;
        updateComputedProps();
      }});

      heightSel.addEventListener("input", (e) => {{
        state.maxalt = Number(e.target.value);
        updateComputedProps();
      }});

      powerSel.addEventListener("input", (e) => {{
        state.power = Number(e.target.value);
        updateComputedProps();
      }});

      document.getElementById("resetBtn").addEventListener("click", () => {{
        state.metric = "population";
        state.yearIndex = YEARS.length - 1;
        state.continent = "All";
        state.language = "All";
        state.maxalt = 0.12;
        state.power = 0.35;

        metricSel.value = state.metric;
        contSel.value = state.continent;
        langSel.value = state.language;
        yearSlider.value = state.yearIndex;
        heightSel.value = state.maxalt;
        powerSel.value = state.power;

        updateComputedProps();
        globe.pointOfView({{ lat: 15, lng: 0, altitude: 2.2 }}, 600);
      }});
    }}

    buildUI();
    updateComputedProps();
  </script>
</head>

<body>
  <div class="brandChip">
    <div class="dot"></div>
    <div class="t">World Globe Dashboard</div>
    <div class="s">drag to orbit · hover for tooltip</div>
  </div>

  <div class="dock">
    <div class="dockTop">
      <div class="pills">
        <div class="pill" id="pillMetric"></div>
        <div class="pill" id="pillYear"></div>
        <div class="pill" id="pillCont"></div>
        <div class="pill" id="pillLang"></div>
        <div class="pill" id="pillH"></div>
        <div class="pill" id="pillP"></div>
      </div>
    </div>

    <div class="dockBody">
      <div class="control">
        <label>Metric</label>
        <select id="metric">
          <option value="population">Population</option>
          <option value="gdp_pc">GDP per capita</option>
        </select>
      </div>

      <div class="control">
        <label>Continent</label>
        <select id="continent">
          <option value="All">All</option>
          <option value="Africa">Africa</option>
          <option value="Asia">Asia</option>
          <option value="Europe">Europe</option>
          <option value="Americas">Americas</option>
          <option value="Oceania">Oceania</option>
        </select>
      </div>

      <div class="control">
        <label>Language</label>
        <select id="language"></select>
      </div>

      <div class="control">
        <label>Year</label>
        <input id="year" type="range" />
      </div>

      <div class="control">
        <label>Height</label>
        <input id="height" type="range" min="0.02" max="0.20" step="0.01" />
      </div>

      <div class="control">
        <label>Contrast</label>
        <input id="power" type="range" min="0.20" max="0.80" step="0.01" />
      </div>

      <div class="control">
        <label>&nbsp;</label>
        <button id="resetBtn">Reset view</button>
      </div>
    </div>
  </div>

  <div id="globe"></div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate globe.html dashboard.")
    parser.add_argument("--out", default="globe.html")
    args = parser.parse_args()

    # Base datasets
    pop = read_population()
    gdp = read_gdp_pc()
    cont = read_continents()

    years = make_years(pop)

    pop_series = build_iso3_series(pop, years, "population")
    gdp = gdp[gdp["year"].isin(years)].copy()
    gdp_series = build_iso3_series(gdp, years, "gdp_pc")

    continent_map = {r["iso3"]: r["continent"] for _, r in cont.iterrows()}
    continent_map = {k.upper(): v for k, v in continent_map.items()}

    # language mapping
    iso3_to_langs, all_languages = read_country_languages()
    print(f"Loaded language list: {len(all_languages)} unique languages")

    geojson = download_geojson()
    geojson = attach_all_properties(
        geojson=geojson,
        years=years,
        pop_series=pop_series,
        gdp_series=gdp_series,
        continent_map=continent_map,
        iso3_to_langs=iso3_to_langs,
    )

    write_html(Path(args.out), geojson, years, all_languages)

    print("\n Done!")
    print(f"  Wrote: {args.out}")
    print("  Run:")
    print("    python -m http.server 8000")
    print(f'    open "http://localhost:8000/{args.out}"\n')


if __name__ == "__main__":
    main()
