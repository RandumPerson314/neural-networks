"""
Builds a daily + hourly dataset for Ontario, 2002-present:
  1. IESO hourly Ontario demand  (reports-public.ieso.ca)
  2. Population-weighted temperature (Open-Meteo ERA5 archive, 12 largest-population centres)
  3. Ontario holidays (ontario_holidays_2002_2026.csv, shipped alongside)
Requires: pip install pandas requests
"""
import io, datetime as dt, time, requests, pandas as pd

START, END = "2002-05-01", dt.date.today().isoformat()  # IESO market data begins May 2002

# ---------- 1. IESO demand ----------
def ieso_year(y):
    url = f"https://reports-public.ieso.ca/public/Demand/PUB_Demand_{y}.csv"
    txt = requests.get(url, timeout=60).text
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Date,Hour"))
    df = pd.read_csv(io.StringIO("\n".join(lines[i:])))
    df["datetime_he"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h")  # hour-ending, EST
    return df

demand = pd.concat([ieso_year(y) for y in range(2002, dt.date.today().year + 1)], ignore_index=True)
demand = demand.rename(columns={"Ontario Demand": "ontario_demand_mw", "Market Demand": "market_demand_mw"})
demand = demand[["datetime_he", "ontario_demand_mw", "market_demand_mw"]].drop_duplicates("datetime_he")

# ---------- 2. Population-weighted temperature ----------
# 2021 Census populations (CMA/city, approximate) -> weights. Edit/extend as desired.
CITIES = {  # name: (lat, lon, population)
    "Toronto":   (43.65, -79.38, 2794356),
    "Ottawa":    (45.42, -75.70, 1017449),
    "Mississauga":(43.59, -79.64, 717961),
    "Brampton":  (43.73, -79.76, 656480),
    "Hamilton":  (43.26, -79.87, 569353),
    "London":    (42.98, -81.25, 422324),
    "Markham":   (43.86, -79.34, 338503),
    "Vaughan":   (43.84, -79.51, 323103),
    "Kitchener": (43.45, -80.49, 256885),
    "Windsor":   (42.32, -83.04, 229660),
    "Sudbury":   (46.49, -80.99, 166004),
    "Thunder Bay":(48.38, -89.25, 108843),
}
tot = sum(p for *_, p in CITIES.values())

def temps(name, lat, lon):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
        latitude=lat, longitude=lon, start_date=START, end_date=END,
        hourly="temperature_2m", timezone="Etc/GMT+5"), timeout=120)  # fixed EST to match IESO
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), name=name)

series = {}
for n, (la, lo, p) in CITIES.items():
    series[n] = temps(n, la, lo); time.sleep(15)   # be polite / avoid rate limit
T = pd.DataFrame(series)
T["temp_popweighted_c"] = sum(T[n] * CITIES[n][2] for n in CITIES) / tot
T.index = T.index + pd.Timedelta(hours=1)           # Open-Meteo is hour-beginning; IESO is hour-ending
T.index.name = "datetime_he"

# ---------- 3. Holidays ----------
hol = pd.read_csv("ontario_holidays_2002_2026.csv", parse_dates=["date"])

# ---------- merge ----------
df = demand.merge(T.reset_index(), on="datetime_he", how="left")
df["date"] = (df["datetime_he"] - pd.Timedelta(hours=1)).dt.normalize()
df = df.merge(hol.rename(columns={"holiday": "holiday_name"}), on="date", how="left")
df["is_holiday"] = df["holiday_name"].notna() & df["statutory"].fillna(False)
df.to_csv("ontario_hourly_demand_temp_holidays.csv", index=False)
print(df.shape, df.datetime_he.min(), df.datetime_he.max())
