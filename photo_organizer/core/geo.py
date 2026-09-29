"""GPS metadata extraction and fast offline reverse geocoding."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

try:
    from PIL import ExifTags
except ImportError:
    ExifTags = None

# Offline reference database of major world cities and tourist destinations
# Format: (city, country, latitude, longitude)
_OFFLINE_CITIES: List[Tuple[str, str, float, float]] = [
    # North America
    ("New York", "United States", 40.7128, -74.0060),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("San Francisco", "United States", 37.7749, -122.4194),
    ("Chicago", "United States", 41.8781, -87.6298),
    ("Miami", "United States", 25.7617, -80.1918),
    ("Seattle", "United States", 47.6062, -122.3321),
    ("Las Vegas", "United States", 36.1699, -115.1398),
    ("Honolulu", "United States", 21.3069, -157.8583),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Montreal", "Canada", 45.5017, -73.5673),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Cancun", "Mexico", 21.1619, -86.8515),

    # Europe
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("Paris", "France", 48.8566, 2.3522),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Venice", "Italy", 45.4408, 12.3155),
    ("Florence", "Italy", 43.7696, 11.2558),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Barcelona", "Spain", 41.3851, 2.1734),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Munich", "Germany", 48.1351, 11.5820),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Vienna", "Austria", 48.2082, 16.3738),
    ("Prague", "Czech Republic", 50.0755, 14.4378),
    ("Zurich", "Switzerland", 47.3769, 8.5417),
    ("Athens", "Greece", 37.9838, 23.7275),
    ("Santorini", "Greece", 36.3932, 25.4615),
    ("Dublin", "Ireland", 53.3498, -6.2603),
    ("Edinburgh", "United Kingdom", 55.9533, -3.1883),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Stockholm", "Sweden", 59.3293, 18.0686),

    # Asia & Middle East
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Abu Dhabi", "United Arab Emirates", 24.4539, 54.3773),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Kyoto", "Japan", 35.0116, 135.7681),
    ("Osaka", "Japan", 34.6937, 135.5023),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Phuket", "Thailand", 7.8804, 98.3923),
    ("Bali", "Indonesia", -8.4095, 115.1889),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Hong Kong", "Hong Kong", 22.3193, 114.1694),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Bengaluru", "India", 12.9716, 77.5946),
    ("Hyderabad", "India", 17.3850, 78.4867),
    ("Chennai", "India", 13.0827, 80.2707),
    ("Kolkata", "India", 22.5726, 88.3639),
    ("Goa", "India", 15.2993, 74.1240),
    ("Jaipur", "India", 26.9124, 75.7873),
    ("Kochi", "India", 9.9312, 76.2673),
    ("Agra", "India", 27.1767, 78.0081),
    ("Varanasi", "India", 25.3176, 82.9739),
    ("Istanbul", "Turkey", 41.0082, 28.9784),

    # Oceania
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Melbourne", "Australia", -37.8136, 144.9631),
    ("Auckland", "New Zealand", -36.8485, 174.7633),
    ("Queenstown", "New Zealand", -45.0312, 168.6626),

    # South America & Africa
    ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Marrakech", "Morocco", 31.6295, -7.9811),
]


def _convert_to_degrees(value) -> Optional[float]:
    """Converts EXIF (degrees, minutes, seconds) tuple to decimal degrees."""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def extract_gps_from_exif(exif) -> Optional[Tuple[float, float]]:
    """Extracts (latitude, longitude) decimal coordinates from a Pillow image EXIF object."""
    if not exif:
        return None

    try:
        # Check standard IFD GPSInfo
        gps_info = None
        if hasattr(exif, "get_ifd") and ExifTags is not None:
            try:
                gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except Exception:
                gps_info = None

        if not gps_info:
            # Fallback to direct tag 0x8825
            gps_info = exif.get(0x8825)

        if not gps_info:
            return None

        # Tags: 1=LatitudeRef, 2=Latitude, 3=LongitudeRef, 4=Longitude
        lat = _convert_to_degrees(gps_info.get(2))
        lat_ref = gps_info.get(1)
        lon = _convert_to_degrees(gps_info.get(4))
        lon_ref = gps_info.get(3)

        if lat is None or lon is None:
            return None

        if lat_ref and str(lat_ref).upper().startswith("S"):
            lat = -lat
        if lon_ref and str(lon_ref).upper().startswith("W"):
            lon = -lon

        return round(lat, 6), round(lon, 6)
    except Exception:
        return None


def reverse_geocode_offline(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    """Finds the closest known city and country using offline Euclidean/Haversine distance."""
    if lat is None or lon is None:
        return None, None

    best_dist = float("inf")
    best_city, best_country = None, None

    for city, country, c_lat, c_lon in _OFFLINE_CITIES:
        # Fast equirectangular approximation
        x = (lon - c_lon) * math.cos(math.radians((lat + c_lat) / 2.0))
        y = lat - c_lat
        dist = math.hypot(x, y)
        if dist < best_dist:
            best_dist = dist
            best_city, best_country = city, country

    # Approximate cutoff ~ 800 km in angular distance
    if best_dist < 8.0:
        return best_city, best_country
    return None, None
