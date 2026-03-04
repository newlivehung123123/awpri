"""
AWPRI Panel Configuration
25 countries x 2004-2022 = 475 country-year observations
"""

COUNTRIES = {
    "AU": ("Australia",    "AUS"),
    "BR": ("Brazil",       "BRA"),
    "CA": ("Canada",       "CAN"),
    "FR": ("France",       "FRA"),
    "DE": ("Germany",      "DEU"),
    "IN": ("India",        "IND"),
    "IT": ("Italy",        "ITA"),
    "JP": ("Japan",        "JPN"),
    "NL": ("Netherlands",  "NLD"),
    "NZ": ("New Zealand",  "NZL"),
    "KR": ("South Korea",  "KOR"),
    "ES": ("Spain",        "ESP"),
    "SE": ("Sweden",       "SWE"),
    "GB": ("United Kingdom","GBR"),
    "US": ("United States","USA"),
    "AR": ("Argentina",    "ARG"),
    "CN": ("China",        "CHN"),
    "DK": ("Denmark",      "DNK"),
    "KE": ("Kenya",        "KEN"),
    "MX": ("Mexico",       "MEX"),
    "NG": ("Nigeria",      "NGA"),
    "PL": ("Poland",       "POL"),
    "ZA": ("South Africa", "ZAF"),
    "TH": ("Thailand",     "THA"),
    "VN": ("Vietnam",      "VNM"),
}

YEARS = list(range(2004, 2023))  # 2004-2022 inclusive

ISO2_LIST = list(COUNTRIES.keys())
ISO3_LIST = [v[1] for v in COUNTRIES.values()]
NAMES     = {iso2: v[0] for iso2, v in COUNTRIES.items()}
ISO2_TO_ISO3 = {iso2: v[1] for iso2, v in COUNTRIES.items()}
ISO3_TO_ISO2 = {v[1]: iso2 for iso2, v in COUNTRIES.items()}

FAO_CODES = {
    "AU": 10,  "BR": 21,  "CA": 33,  "FR": 68,  "DE": 79,
    "IN": 100, "IT": 106, "JP": 110, "NL": 156, "NZ": 162,
    "KR": 116, "ES": 203, "SE": 210, "GB": 229, "US": 231,
    "AR": 9,   "CN": 351, "DK": 58,  "KE": 114, "MX": 138,
    "NG": 159, "PL": 173, "ZA": 202, "TH": 216, "VN": 237,
}

POP_MILLIONS = {
    "AU": 25.98,  "BR": 215.31, "CA": 38.25,  "FR": 67.90,  "DE": 83.79,
    "IN": 1417.17,"IT": 60.32,  "JP": 125.12, "NL": 17.62,  "NZ": 5.12,
    "KR": 51.74,  "ES": 47.43,  "SE": 10.55,  "GB": 67.51,  "US": 333.29,
    "AR": 45.20,  "CN": 1425.67,"DK": 5.91,   "KE": 53.01,  "MX": 127.50,
    "NG": 218.54, "PL": 37.84,  "ZA": 59.89,  "TH": 71.70,  "VN": 97.34,
}

GDP_BILLIONS = {
    "AU": 1703.0, "BR": 1920.1, "CA": 2140.0, "FR": 2782.9, "DE": 4072.2,
    "IN": 3385.1, "IT": 2010.0, "JP": 4231.1, "NL": 1011.0, "NZ": 247.3,
    "KR": 1665.2, "ES": 1418.7, "SE": 585.9,  "GB": 3070.7, "US": 25462.7,
    "AR": 630.5,  "CN": 17963.2,"DK": 395.1,  "KE": 113.4,  "MX": 1322.5,
    "NG": 477.4,  "PL": 688.2,  "ZA": 405.9,  "TH": 495.9,  "VN": 408.9,
}

VDEM_PATH = "data/raw/V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.csv"
