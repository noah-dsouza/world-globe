# 🌍 World Globe Dashboard

Interactive 3D visualization of global socioeconomic data at the country level.

This project renders a fully rotatable globe where countries can be filtered and extruded by **population**, **GDP per capita**, and **languages spoken**, with live interaction and no page reloads.

---

## ✨ Features

- 🌐 Interactive 3D globe with full rotation, zoom, and tilt  
- 📊 Country-level height extrusion by:
  - Population (1960–2023)
  - GDP per capita
  - Languages spoken
- 🗺️ Filters for:
  - Continent
  - Language spoken
- ⚡ Real-time updates via UI controls  
- 🏷️ Country tooltips with metric values  
- 🎯 Visual dimming of non-matching countries  
- 🧼 Clean, dock-style control panel  

---

## 🗂️ Project Structure
world-globe/
├── main.py # Data processing + HTML generator
├── globe.html # Generated visualization
├── venv/ # Python virtual environment
└── data/
├── World Population 1960-2023 by Country.csv
├── API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2.csv
├── continents2.csv
└── countries-languages-spoken.csv


---

## ⚙️ Setup

### 📦 Requirements
- Python 3.10+
- macOS / Linux

### 🔧 Installation

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install pandas numpy requests pycountry

python main.py
python -m http.server 8000
http://localhost:8000/globe.html

