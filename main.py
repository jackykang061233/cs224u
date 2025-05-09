from core.app import FunctionCallingAgent
from core.google_maps_api import find_places_within_travel_distance
from core.help_functions.ranking import rank_places

    
if __name__ == '__main__':
    # "Can you recommend a good coffee shop near downtown Boston that’s within a 10-minute walk?",
    #     "I’m looking for a highly-rated Thai restaurant in San Francisco—any suggestions?",
    user_querys = [
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

    agent = FunctionCallingAgent()
    for user_query in user_querys:
        print(user_query)
        result = agent.query(user_query=user_query)
        print(result)
        
        places = find_places_within_travel_distance(location=result['validated']['location']['coordinates'],
                                           minimum_star_requirement=result['validated']['minimum_star_requirement']['rating'],
                                           place_type=result['validated']['place_to_search'],
                                           travel_mode=result['validated']['travel_duration']['mode'],
                                           max_travel_time=result['validated']['travel_duration']['value'])
        if places:
            # for place in places:
            #     print(place)
            # input(123)
            df = rank_places(places, keywords=result['validated']['additional_requests'])
            print(df)
        else:
            print('No recommend places')
        input('next')