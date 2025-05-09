from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from fuzzywuzzy import process, fuzz
from typing import Dict, List, Optional, Tuple
import time
import json

class LocationError(Exception):
    pass

def disambiguate_location(
    query: str,
    geolocation_coords: Optional[Tuple[float, float]] = None,
    max_results: int = 5,
    retries: int = 2,
    timeout: int = 5
) -> Dict:
    """
    Resolve an ambiguous location query, allowing user selection from multiple results.
    Args:
        query: Location string (e.g., "Paris", "Pariis", "Springfield").
        geolocation_coords: User's current coordinates for prioritizing results.
        max_results: Maximum number of results to return.
        retries: Number of retries for transient errors.
        timeout: Geocoding request timeout in seconds.
    Returns:
        Dict with type, value, coordinates, and options (if ambiguous), or raises LocationError.
    """
    # Initialize geocoder
    geolocator = Nominatim(user_agent="place_recommendation_app")

    # Reference locations for fuzzy matching (can be database-driven)
    reference_locations = [
        {"name": "Paris", "type": "city", "country": "France", "coordinates": (48.8566, 2.3522)},
        {"name": "New York City", "type": "city", "country": "USA", "coordinates": (40.7128, -74.0060)},
    ]
    aliases = {"The Big Apple": "New York City", "NYC": "New York City"}

    # Handle empty query
    if not query:
        return {"type": None, "value": None, "coordinates": None, "options": None}

    # Handle "near me"
    if query.lower() == "near me":
        query = 'New York City'
        # if not geolocation_coords:
        #     raise LocationError("No geolocation provided for 'near me'")
        # return {
        #     "type": "near me",
        #     "value": "Current Location",
        #     "coordinates": geolocation_coords,
        #     "options": None
        # }

    # Check aliases
    query = aliases.get(query.lower(), query)

    # Fuzzy matching
    location_names = [loc["name"] for loc in reference_locations]
    best_match = process.extractOne(
        query,
        location_names,
        scorer=fuzz.token_set_ratio,
        score_cutoff=80
    )

    if best_match:
        matched_loc = next(loc for loc in reference_locations if loc["name"] == best_match[0])
        result = {
            "type": matched_loc["type"],
            "value": matched_loc["name"],
            "coordinates": matched_loc["coordinates"],
            "options": None
        }
        return result

    # Geocoding with retries
    attempt = 0
    while attempt <= retries:
        try:
            locations = geolocator.geocode(
                query,
                exactly_one=False,
                limit=max_results,
                timeout=timeout
            )
            break
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            attempt += 1
            if attempt > retries:
                raise LocationError(f"Geocoding failed after {retries} retries: {str(e)}")
            time.sleep(1)  # Respect Nominatim rate limits

    if not locations:
        raise LocationError(f"No results found for '{query}'")

    # Single result
    if len(locations) == 1:
        loc = locations[0]
        loc_type = "city" if "city" in loc.raw.get("type", "") else "landmark"
        result = {
            "type": loc_type,
            "value": loc.address,
            "coordinates": (loc.latitude, loc.longitude),
            "options": None
        }
        return result

    # Multiple results: Prioritize by geolocation (if provided)
    options = [
        {
            "type": "city" if "city" in loc.raw.get("type", "") else "landmark",
            "value": loc.address,
            "coordinates": (loc.latitude, loc.longitude)
        }
        for loc in locations
    ]

    if geolocation_coords:
        def distance_to_user(option):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371  # Earth's radius in km
            lat1, lon1 = geolocation_coords
            lat2, lon2 = option["coordinates"]
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c  # Distance in km

        options.sort(key=distance_to_user)

    return {
        "type": None,
        "value": None,
        "coordinates": None,
        "options": options  # Caller handles user selection
    }
    
if __name__ == '__main__':
    result = disambiguate_location('near me')
    for key, item in result.items():
        print(key)
        print(item)
    for option in result['options']:
        print(option)
    # # location = (result.latitude, result.longitude)
    # print(location)