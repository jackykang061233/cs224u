import googlemaps
from datetime import datetime
from dotenv import load_dotenv
import os
from core.help_functions.detecting_location import disambiguate_location
import time
 
load_dotenv()
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
 
def get_travel_time(origin, destinations, travel_mode='walking', chunk_size=25):
    """Calculate travel time from origin to multiple destinations in chunks for the specified mode."""
    valid_modes = ['walking', 'driving', 'transit', 'bicycling']
    if travel_mode.lower() not in valid_modes:
        raise ValueError(f"Invalid travel mode: {travel_mode}. Must be one of {valid_modes}.")
   
    times = []
    for i in range(0, len(destinations), chunk_size):
        chunk = destinations[i:i+chunk_size]
        try:
            result = gmaps.distance_matrix(
                origins=[origin],
                destinations=chunk,
                mode=travel_mode.lower(),
                units='metric'
            )
            for element in result['rows'][0]['elements']:
                if element['status'] == 'OK':
                    times.append(element['duration']['value'])  # Time in seconds
                else:
                    times.append(float('inf'))  # Handle cases where no route is found
        except Exception as e:
            print(f"Error calculating travel time for chunk {i // chunk_size + 1}: {e}")
            times.extend([float('inf')] * len(chunk))
    return times
 
def get_place_details(place_id):
    """Fetch detailed information for a place using its place_id, sorting reviews by recency."""
    try:
        details = gmaps.place(
            place_id=place_id,
            fields=[
                'name', 'formatted_address', 'rating', 'reviews',
                'formatted_phone_number', 'website', 'opening_hours', 'geometry/location',
                'price_level'
            ]
        )
        result = details.get('result', {})
        reviews = result.get('reviews', [])
       
        # Sort reviews by recency using Unix timestamp
        sorted_reviews = sorted(
            reviews,
            key=lambda x: x.get('time', 0),
            reverse=True  # Most recent first
        )[:5]  # Ensure max 5 reviews
       
        return {
            'name': result.get('name'),
            'address': result.get('formatted_address'),
            'rating': result.get('rating', 0),
            'price_level': result.get('price_level', 'N/A'),  # 0-4, N/A if not available
            'reviews': [
                {
                    'author': review.get('author_name'),
                    'rating': review.get('rating'),
                    'text': review.get('text'),
                    'time': review.get('relative_time_description', 'Unknown time'),
                    'timestamp': review.get('time', 0)
                } for review in sorted_reviews
            ],
            'phone': result.get('formatted_phone_number', 'N/A'),
            'website': result.get('website', 'N/A'),
            'opening_hours': result.get('opening_hours', {}).get('weekday_text', ['N/A']),
            'location': (
                result.get('geometry', {}).get('location', {}).get('lat'),
                result.get('geometry', {}).get('location', {}).get('lng')
            ),
            'review_note': 'Note: Google Places API returns up to 5 reviews per place (latest selected).'
        }
    except Exception as e:
        print(f"Error fetching details for place_id {place_id}: {e}")
        return None
 
def find_places_within_travel_distance(
    location,
    minimum_star_requirement=3.5,
    place_type='restaurant',
    travel_mode='walking',
    max_travel_time=900,
    open_now=True,
):
    """
    Find places within a specified travel time from a location, with filters for cuisine, price, Wi-Fi, and open status.
   
    Args:
        location: Tuple of (latitude, longitude) or string address
        place_type: Type of place (e.g., 'restaurant', 'store')
        travel_mode: Travel mode ('walking', 'driving', 'transit', 'bicycling')
        max_travel_time: Maximum travel time in seconds (default: 900 for 15 minutes)
        radius: Search radius in meters (default: 1500, approximate for 15-min walk)
        cuisine: Cuisine type (e.g., 'Mexican') for keyword filtering
        price_level: 'cheap' (0-1), 'moderate' (2), 'expensive' (3-4), or None
        free_wifi: If True, search for places with 'free Wi-Fi' in description/reviews
        open_now: If True, only return places currently open
   
    Returns:
        List of places with detailed information
    """
    # Convert address to coordinates if necessary
    if isinstance(location, str):
        geocode = gmaps.geocode(location)
        if not geocode:
            print("Invalid location")
            return []
        location = (geocode[0]['geometry']['location']['lat'],
                    geocode[0]['geometry']['location']['lng'])

 
    # Build keyword for filtering
    keywords = []

    keyword = ' '.join(keywords) if keywords else None
 
    places = []
    next_page_token = None
    
    # Set radius
    if travel_mode == 'walking':
        radius = 1.39*max_travel_time
    elif travel_mode == 'driving':
        radius = 16*max_travel_time
    if travel_mode == 'transit':
        radius = 9*max_travel_time
    if travel_mode == 'bicycling':
        radius = 5*max_travel_time
    
    # Handle pagination (up to 60 results)
    for _ in range(3):  # Max 3 pages (20 results each)
        try:
            response = gmaps.places_nearby(
                location=location,
                radius=radius,
                type=place_type,
                keyword=keyword,
                open_now=open_now,
                page_token=next_page_token
            )
 
            for place in response.get('results', []):
                places.append({
                    'place_id': place.get('place_id'),
                    'location': (
                        place['geometry']['location']['lat'],
                        place['geometry']['location']['lng']
                    )
                })
 
            next_page_token = response.get('next_page_token')
            if not next_page_token:
                break
 
            time.sleep(2)
 
        except Exception as e:
            print(f"Error fetching places: {e}")
            break

    # Filter places by travel time
    filtered_places = []
    if places:
        destinations = [place['location'] for place in places]
        travel_times = get_travel_time(location, destinations, travel_mode=travel_mode)

        for place, travel_time in zip(places, travel_times):
            # print(place)
            # print(travel_time)
            # print(max_travel_time)
            if travel_time <= max_travel_time:
                details = get_place_details(place['place_id'])
                if details and details['rating'] >= minimum_star_requirement:
                    details['travel_time'] = round(travel_time / 60, 1)
                    details['travel_mode'] = travel_mode.lower()
                    filtered_places.append(details)

    return filtered_places
 
# Example usage
if __name__ == "__main__":
    # Example location: Empire State Building, New York
    result = disambiguate_location('Union Square')
    location = (result.latitude, result.longitude)
    print(location)
    place_type = 'restaurant'  # or 'store' for shops
    travel_mode = 'driving'  # Options: 'walking', 'driving', 'transit', 'bicycling'
    open_now = True  # Only return places currently open
 
    results = find_places_within_travel_distance(
        location=location,
        place_type=place_type,
        travel_mode=travel_mode,
        open_now=open_now
    )
 
    # Print results
    if results:
        print(f"Found {len(results)} places within 15 minutes by {travel_mode}:")
        for place in results:
            print(f"\n- {place['name']}")
            print(f"  Address: {place['address']}")
            print(f"  Travel time: {place['travel_time_minutes']} minutes by {place['travel_mode']}")
            print(f"  Rating: {place['rating']} / 5")
            print(f"  Price Level: {place['price_level']} / 4")
            print(f"  Phone: {place['phone']}")
            print(f"  Website: {place['website']}")
            print(f"  Opening Hours: {', '.join(place['opening_hours'])}")
            print(f"  {place['review_note']}")

            if place['reviews']:
                print("  Reviews (up to 5, sorted by recency):")
                for review in place['reviews']:
                    print(f"    - {review['author']} ({review['rating']}/5, {review['time']}): {review['text'][:100]}...")
            else:
                print("  Reviews: None")
    else:
        print(f"No {place_type}s found within 15 minutes by {travel_mode}.")
 