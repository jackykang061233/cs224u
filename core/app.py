from core.help_functions.LLMClient import LLMClient
from core.help_functions.detecting_location import disambiguate_location
from core.error.errors import TravelDurationError, StarRequirementError, LocationError

from datetime import datetime
import pytz
import json
import yaml
import os

from typing import Dict, Optional, Union, Tuple
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    def __init__(self):
        self.client = LLMClient(api_key=os.getenv('OPENAI_API_KEY'))
        with open('lib/config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        json_format = json.dumps({key: item['type'] for key, item in self.config.items()}, indent=2)
        detail_instruction = '\n'.join([str(index)+'. '+'**'+key+'** '+item["description"]+'\n'+item["prompt"] for index, (key, item) in enumerate(self.config.items(), start=1)])
        example_output = json.dumps({key: item['example'] for key, item in self.config.items()}, indent=2)
        self.prompt = f"""You are a highly accurate information extraction assistant. Your task is to extract specific details from a user's query regarding a local search. You will receive a user query as input and must return a JSON object containing the extracted information.  If a particular piece of information is *not* present in the query, its corresponding value in the JSON should be `null` (not "None").  
**Here's the JSON format you *must* use:**
```json
{json_format}
```
 
**Detailed Instructions for Each Field:**
{detail_instruction}

**Important Considerations:**
* **JSON Output:**  Your *only* output should be a valid JSON object adhering to the specified format.  Do not include any introductory text or explanations.
* **Error Handling:** Assume the input will be imperfect.  Focus on extracting what *is* present, and setting missing values to `null`.
* **Prioritization:** If information appears ambiguous, prioritize extraction based on the most likely intent.
* **Data Types:** Ensure numerical values are extracted as numbers (not strings).
 
**Example Input:**
"Find a highly-rated pizza restaurant near me with at least 4 stars within a 15-minute walk."
 
**Example Output:**
```json
{example_output}
```
 
**Now, process the following user query:**
"""

    def query(self, user_query: str, geolocation_coords: Optional[Tuple[float, float]] = None, context: Optional[Dict] = None) -> Dict:
        """
        Process a user query by calling the LLM and validating extracted fields, handling ambiguous locations via LLM UI.
        Args:
            user_query: User input (e.g., "Find me a rooftop bar in Springfield with 4 stars").
            geolocation_coords: User's current coordinates for "near me" (e.g., (40.7128, -74.0060)).
            context: Optional context for disambiguation (e.g., previous LLM output, location options).
        Returns:
            Dict with status ("success", "ambiguous", "error", "prompt") and validated fields, options, or errors.
        """
        # Initialize context if not provided
        context = context or {}

        # Check if this is a follow-up query for location disambiguation
        if context.get("is_disambiguation"):
            return self._handle_location_disambiguation(user_query, context)

        # Call LLM to extract fields
        try:
            llm_result = self.client.call_llm(
                system_prompt="You are a good assistant",
                user_prompt=self.prompt + user_query,
            )
            llm_result = self.safe_parse_json(llm_result)
        except Exception as e:
            # logging.error(f"LLM call failed: {e}")
            return {
                "status": "error",
                "error": "Failed to process query. Please try again.",
                "prompt": "Please rephrase your query and try again."
            }

        # Expected fields from LLM
        extracted = {
            "location": llm_result.get("location"),
            "place_to_search": llm_result.get("place_to_search"),
            "travel_duration": llm_result.get("travel_duration"),
            "minimum_star_requirement": llm_result.get("minimum_star_requirement"),
            "additional_requests": llm_result.get("additional_requests")
        }

        # Initialize result and errors
        validated = {}
        errors = []
        ambiguous = {}

        # 1. Validate location
        try:
            location_result = self._check_location(extracted["location"], geolocation_coords)
            if location_result.get("options"):  # Ambiguous location
                ambiguous["location"] = location_result["options"]
                # Store context for disambiguation
                context = {
                    "is_disambiguation": True,
                    "original_query": user_query,
                    "extracted": extracted,
                    "location_options": location_result["options"],
                    "geolocation_coords": geolocation_coords
                }
                # Generate LLM prompt for user selection
                options_text = "\n".join(
                    f"{idx + 1}. {opt['value']}" for idx, opt in enumerate(location_result["options"])
                )
                prompt = (
                    f"Multiple locations found for '{extracted['location']}':\n"
                    f"{options_text}\n"
                    "Please select the correct location by entering the number or the location name (e.g., '1' or 'Springfield, IL'). "
                    "Enter 'cancel' to cancel."
                )
                return {
                    "status": "prompt",
                    "prompt": prompt,
                    "context": context
                }
            else:
                validated["location"] = location_result
        except LocationError as e:
            errors.append(f"Location error: {e}")
            validated["location"] = None

        # 2. Validate place_to_search
        validated["place_to_search"] = extracted["place_to_search"]

        # 3. Validate travel_duration
        try:
            travel_result = self._check_travel_duration(extracted["travel_duration"])
            validated["travel_duration"] = travel_result
        except TravelDurationError as e:
            errors.append(f"Travel duration error: {e}")
            validated["travel_duration"] = None

        # 4. Validate minimum_star_requirement
        try:
            star_result = self._check_minimum_star_requirement(
                extracted["minimum_star_requirement"],
                default_rating=3.5  # Optional: Default for null
            )
            validated["minimum_star_requirement"] = star_result
        except StarRequirementError as e:
            errors.append(f"Star requirement error: {e}")
            validated["minimum_star_requirement"] = None

        # 5. Validate additional_requests
        validated["additional_requests"] = extracted["additional_requests"]

        # Determine response status
        if errors:
            return {
                "status": "error",
                "errors": errors,
                "prompt": "Please clarify the following issues:\n" + "\n".join(errors)
            }
        else:
            return {
                "status": "success",
                "validated": validated
            }

    def _handle_location_disambiguation(self, user_response: str, context: Dict) -> Dict:
        """
        Process user response for location disambiguation.
        Args:
            user_response: User's selection (e.g., "1", "Springfield, IL", "cancel").
            context: Context from original query (original_query, extracted, location_options, geolocation_coords).
        Returns:
            Dict with status and validated fields, errors, or further prompts.
        """
        options = context["location_options"]
        user_response = user_response.strip().lower()

        # Handle cancellation
        if user_response == "cancel":
            return {
                "status": "error",
                "error": "Location selection cancelled.",
                "prompt": "Please provide a new query or specify a precise location."
            }

        # Parse response (number or name)
        selected_option = None
        try:
            # Try as index
            index = int(user_response) - 1
            if 0 <= index < len(options):
                selected_option = options[index]
        except ValueError:
            # Try as location name
            for opt in options:
                if user_response == opt["value"].lower() or user_response in opt["value"].lower():
                    selected_option = opt
                    break

        if not selected_option:
            options_text = "\n".join(f"{idx + 1}. {opt['value']}" for idx, opt in enumerate(options))
            prompt = (
                f"Invalid selection: '{user_response}'. Please select a valid location by entering the number or name:\n"
                f"{options_text}\n"
                "Enter 'cancel' to cancel."
            )
            return {
                "status": "prompt",
                "prompt": prompt,
                "context": context
            }

        # Update extracted fields with selected location
        extracted = context["extracted"]
        extracted["location"] = selected_option["value"]

        # Re-run query with updated location
        return self.query(
            user_query=context["original_query"],
            geolocation_coords=context["geolocation_coords"],
            context={"extracted": extracted}  # Pass updated extracted fields
        )

    def _check_location(self, location: str, geolocation_coords: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Validate and resolve a location string to coordinates and metadata.
        Args:
            location: User-provided location (e.g., "Paris", "near me", "Eifel Tower").
            geolocation_coords: User's current coordinates (from geolocation system).
        Returns:
            Dict with type, value, coordinates, or raises LocationError.
        """
        if not location:
            return {"type": None, "value": None, "coordinates": None, "options": None}

        # Call disambiguate_location
        result = disambiguate_location(
            query=location,
            geolocation_coords=geolocation_coords,
            cache=None
        )

        if result["options"]:
            # Ambiguous location: Pass options to the app for user selection
            return result
        elif result["type"]:
            # Resolved location
            return result
        else:
            raise LocationError(f"Could not resolve location: {location}")
    
    def _check_travel_duration(self, travel_duration: Optional[Dict]) -> Dict:
        """
        Validate and process travel duration.
        Args:
            travel_duration: Dict with value, unit, mode (e.g., {'value': 10, 'unit': 'minutes', 'mode': 'walking'}).
        Returns:
            Dict with validated value, unit, mode, or raises TravelDurationError.
        """
        valid_modes = {"walking", "driving", "bicycling", "transit"}
        default_durations = {
            "walking": 15,  # minutes
            "driving": 10,
            "bicycling": 20,
            "transit": 30
        }

        if not travel_duration:
            return {"value": 15, "unit": "minutes", "mode": "walking"}

        mode = travel_duration.get("mode")
        value = travel_duration.get("value")
        unit = travel_duration.get("unit")

        # Validate mode
        if mode and mode.lower() not in valid_modes:
            raise TravelDurationError(f"Invalid travel mode: {mode}. Must be one of {valid_modes}")

        # Handle missing mode
        if not mode:
            return {"value": 15, "unit": "minutes", "mode": "walking"}

        # Handle vague phrases (value and unit missing)
        if not value and not unit:
            return {
                "value": default_durations[mode.lower()],
                "unit": "minutes",
                "mode": mode.lower()
            }

        # Validate value
        try:
            value = float(value)
            if value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise TravelDurationError(f"Invalid travel duration value: {value}. Must be a positive number")

        # Handle missing or invalid unit
        if not unit:
            unit = "minutes"  # Default unit
        elif unit.lower() not in {"minutes", "hours", "seconds"}:
            raise TravelDurationError(f"Invalid unit: {unit}. Must be 'minutes', 'hours', or 'seconds'")

        # Convert to minutes for API consistency
        if unit.lower() == "hours":
            value *= 60
            unit = "minutes"
        elif unit.lower() == "seconds":
            value /= 60
            unit = "minutes"

        return {
            "value": value,
            "unit": "minutes",
            "mode": mode.lower()
        }
        
        
    def _check_minimum_star_requirement(
        self,
        stars: Union[str, float, None],
        fuzzy_tolerance: float = 0.1,
        default_rating: Optional[float] = None
    ) -> Dict:
        """
        Validate and process minimum star requirement from LLM output.
        Args:
            stars: Minimum rating (e.g., 4.0, "highly-rated", "fantastic", None).
            fuzzy_tolerance: Tolerance for fuzzy filtering (e.g., 0.1 allows 3.9 for 4.0).
            default_rating: Optional default rating if None (e.g., 3.5); if unset, returns None.
        Returns:
            Dict with validated rating and fuzzy_rating, or raises StarRequirementError.
        """
        qualitative_mappings = {
            "excellent": 4.5,
            "amazing": 4.5,
            "outstanding": 4.5,
            "highly-rated": 4.0,
            "top-rated": 4.0,
            "best": 4.0,
            "good": 3.5,
            "decent": 3.5,
            "average": 3.0,
            "okay": 3.0
        }

        # Handle null or unspecified input
        if stars is None:
            return {
                "rating": default_rating,
                "fuzzy_rating": default_rating - fuzzy_tolerance if default_rating is not None else None
            }

        # Handle qualitative terms
        if isinstance(stars, str):
            stars_lower = stars.lower()
            if stars_lower in qualitative_mappings:
                rating = qualitative_mappings[stars_lower]
                return {
                    "rating": rating,
                    "fuzzy_rating": max(0.0, rating - fuzzy_tolerance)
                }
            raise StarRequirementError(
                f"Unrecognized qualitative term: {stars}. Valid terms: {list(qualitative_mappings.keys())}"
            )

        # Handle numerical input
        try:
            rating = float(stars)
            rating = round(rating, 1)  # Round to one decimal for API compatibility
            if not 0.0 <= rating <= 5.0:
                raise ValueError
            return {
                "rating": rating,
                "fuzzy_rating": max(0.0, rating - fuzzy_tolerance)
            }
        except (TypeError, ValueError):
            raise StarRequirementError(
                f"Invalid star requirement: {stars}. Must be a number between 0.0 and 5.0 or a valid qualitative term "
                f"({list(qualitative_mappings.keys())})"
            )
    def safe_parse_json(self, text):
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.lstrip("`").strip()
            if clean_text.lower().startswith("json"):
                clean_text = clean_text[4:].strip()
            clean_text = clean_text.rstrip("`").strip()
        return json.loads(clean_text)
 



if __name__ == '__main__':

    agent = MainAgent()
    
    user_querys = ["Can you recommend a good coffee shop near downtown Boston that’s within a 10-minute walk?",
    "I’m looking for a highly-rated Thai restaurant in San Francisco—any suggestions?",
    "What’s the best bakery within walking distance from my hotel in Paris?",
    "Are there any nice parks in Toronto that are less than 15 minutes away on foot?",
    "Find me a rooftop bar in New York City with at least 4 stars on Google Maps.",
    "Any suggestions for a quiet place to read near Central Park?",
    "I want to find a pet-friendly café close to my location—what do you recommend?",
    "Could you help me find a cozy bookstore within a 20-minute walk from here?",
    "What are some top-rated Italian restaurants near me with outdoor seating?",
    "Are there any museums nearby that are highly rated and open today?",
    "Suggest a good spot for brunch in Seattle that’s not too far to walk to.",
    "I'm in Chicago—where should I go for dessert that has at least 4.5 stars?",
    "I’m near Union Square. Know of any vegan restaurants with good reviews?",
    "Where can I find a dog park that’s close and well-rated?",
    "Looking for a bar with live music within a 10-minute walk—any ideas?",
    "Can you suggest a family-friendly attraction near my Airbnb?",
    "I need a quiet café to work from in Berlin, preferably within 15 minutes walking distance.",
    "What’s a good place for a quick lunch around this area with solid reviews?",
    "Find me a sushi place nearby with at least 4 stars and good lunch specials.",
    "Are there any scenic spots or viewpoints I can easily walk to from here?",
    ]
    
    for user_query in user_querys:
        print(user_query)
        result = agent.query(user_query=user_query)
        print(result)
        input('next')






    # prompt = '''You are a highly accurate information extraction assistant. Your task is to extract specific details from a user's query regarding a local search. You will receive a user query as input and must return a JSON object containing the extracted information.  If a particular piece of information is *not* present in the query, its corresponding value in the JSON should be `null` (not "None").  
    # **Here's the JSON format you *must* use:**
    # ```json
    # {
    #   "location": "string or null",
    #   "place_to_search": "string or null",
    #   "travel_duration": {
    #     "value": "number or null",
    #     "unit": "string (e.g., 'minutes', 'hours', 'seconds') or null"
    #     mode: "string or null"
    #   },
    #   "minimum_star_requirement": "number or null",
    #   "additional_requests": "list or null"
    # }
    # ```
    
    # **Detailed Instructions for Each Field:**
    # 1. **location:**  The user's current or desired location. This can be:
    #    * A country (e.g., "France")
    #    * A city (e.g., "Paris")
    #    * A specific address (e.g., "1600 Amphitheatre Parkway, Mountain View, CA")
    #    * A landmark (e.g., "Eiffel Tower")
    #    * GPS coordinates (e.g., "37.7749, -122.4194")
    #    * A relative location (e.g., "near me"). If "near me" is used, consider it as a signal that you need to detect other parameters to determine the location (eg. city name in the query)
    #    * If no location is mentioned, set to `null`.
    
    # 2. **place_to_search:** The type of place the user is looking for. Examples:
    #    * "restaurant"
    #    * "coffee shop"
    #    * "grocery store"
    #    * "pharmacy"
    #    * "shopping mall"
    #    * "bookstore"
    #    * Be as specific as possible.  If the user says "a place to eat", use "restaurant".  If they say "food", also use "restaurant."
    #    * If no place is specified, set to `null`.
    
    # 3. **travel_duration**: The maximum acceptable travel time from the user's location.
    #     * value: A numerical value representing the duration.
    #     * unit: The unit of time (e.g., "minutes", "hours", "seconds"). Be consistent with units.
    #     * mode: The mode of travel, which can be "driving", "walking", "bicycling", or "transit". Default to "walking" if not specified.
    #     * If an explicit duration is provided, extract the numerical value, unit, and mode (e.g., "within 10 minutes by walking" → value: 10, unit: "minutes", mode: "walking").
    #     * For vague phrases indicating proximity by a specific mode (e.g., "within walking distance", "a short drive", "biking distance", "accessible by transit"):
    #       * Infer a default duration based on the mode:
    #         * Walking: Use 15 as the value and "minutes" as the unit for phrases like "within walking distance" or "walkable".
    #         * Driving: Use 10 as the value and "minutes" as the unit for phrases like "a short drive" or "driving distance".
    #         * Bicycling: Use 20 as the value and "minutes" as the unit for phrases like "biking distance" or "bikeable".
    #         * Transit: Use 30 as the value and "minutes" as the unit for phrases like "accessible by transit" or "by public transport".
    #     * If no duration or vague phrase is specified, set value, unit, and mode to null.
    #     * Use the maps_distance_matrix function to calculate distances and durations between origins and destinations with the specified mode.
    #     * Examples:
    #       * "within 10 minutes by walking" → value: 10, unit: "minutes", mode: "walking"
    #       * "a 2-hour drive" → value: 2, unit: "hours", mode: "driving"
    #       * "within walking distance" → value: 15, unit: "minutes", mode: "walking"
    #       * "a short drive" → value: 10, unit: "minutes", mode: "driving"
    #       * "biking distance" → value: 20, unit: "minutes", mode: "bicycling"
    #       * "by transit in 40 minutes" → value: 40, unit: "minutes", mode: "transit"
    
    # 4. **minimum_star_requirement:** The minimum Google Maps star rating the place should have.
    #    * Extract the numerical value if explicitly mentioned. Ignore phrases like "at least", "minimum", etc.
    #    * For qualitative terms indicating high quality (e.g., "highly-rated", "top-rated", "best"), infer a reasonable numerical star rating:
    #      * Use 4.0 for terms like "highly-rated", "top-rated", or "best".
    #      * Use 3.5 for terms like "good" or "decent".
    #    * If no star rating or qualitative term is specified, set to null.
    #    * Examples:
    #      * "4 stars or higher" → 4.0
    #      * "minimum 3.5 stars" → 3.5
    #      * "highly-rated" → 4.0
    #      * "good restaurants" → 3.5
    
    # 5. **additional_requests:** Any other relevant information or constraints the user mentions. It could be more than one. This could include:
    #    * Specific features (e.g., "with outdoor seating", "open now", "dog-friendly")
    #    * Price range (e.g., "cheap", "expensive")
    #    * Cuisine type (e.g., "Italian", "Mexican")
    #    * Amenities (e.g., "free Wi-Fi", "parking")
    #    * If there are no additional requests, set to `null`.  Don't include generic phrases like "please" or "thank you."
    
    # **Important Considerations:**
    # * **JSON Output:**  Your *only* output should be a valid JSON object adhering to the specified format.  Do not include any introductory text or explanations.
    # * **Error Handling:** Assume the input will be imperfect.  Focus on extracting what *is* present, and setting missing values to `null`.
    # * **Prioritization:** If information appears ambiguous, prioritize extraction based on the most likely intent.
    # * **Data Types:** Ensure numerical values are extracted as numbers (not strings).
    
    # **Example Input:**
    # "Find a highly-rated pizza restaurant near me with at least 4 stars within a 15-minute walk."
    
    # **Example Output:**
    # ```json
    # {
    #   "location": "near me",
    #   "place_to_search": "restaurant",
    #   "travel_duration": {
    #     "value": 15,
    #     "unit": "minutes"
    #   },
    #   "minimum_star_requirement": 4,
    #   "additional_requests": ["highly-rated", "pizza"]
    # }
    # ```
    
    # **Now, process the following user query:**
    # '''
    
