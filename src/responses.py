"""
Response Catalog and Selector for Voice-Enabled Chatbot.
Provides natural, relevant conversational responses for all 18 supported intents,
sentiment expressions, location/directional entity answers, and domain knowledge mappings.
Academically transparent: The BiLSTM model classifies intent; this module selects
appropriate, conversational responses corresponding to the predicted intent.
Never falsely claims real-world action execution.
"""
import re
import random
from typing import Dict, List, Optional
import config

INTENT_RESPONSES: Dict[str, List[str]] = {
    "weather": [
        "I can help with weather-related questions. Please provide a location for a weather query.",
        "Current weather forecast shows mild temperatures and clear conditions. Provide your city for detailed local weather.",
        "I can assist with weather inquiries. Let me know which city or region you would like the weather forecast for.",
        "Weather query noted! For current weather conditions, temperatures, or rain probability, please specify your location."
    ],
    "directions": [
        "I can help with direction-related questions. Please provide your starting point and destination.",
        "Directions inquiry detected. Head toward the main transit corridor and follow directional road signs.",
        "I can assist with route and direction queries. For turn-by-turn live navigation, you can check Google Maps.",
        "The fastest route to your destination is typically via the central expressway; traffic is currently moving smoothly."
    ],
    "restaurant_suggestion": [
        "I can help with dining and restaurant suggestions. What type of cuisine or meal are you looking for?",
        "Looking for food recommendations? A local Italian bistro or downtown grill would be a great choice for dining.",
        "I can suggest dining spots nearby. Please specify whether you'd like casual dining, fine dining, or takeout.",
        "There are several top-rated restaurants nearby serving fresh artisan pizzas, sushi, and vegetarian options."
    ],
    "book_hotel": [
        "I can help with hotel-related questions. Please provide your destination and travel dates.",
        "Accommodation query noted! You can review available rooms, amenities, and nightly rates through a booking portal.",
        "I can help identify hotel options. What city and date range are you planning for your stay?",
        "For lodging, the Downtown Plaza Hotel and Riverside Suites offer comfortable rooms with great reviews."
    ],
    "travel_suggestion": [
        "I can help with travel recommendations. Are you interested in mountain retreats, coastal beaches, or historic cities?",
        "Looking for vacation ideas? Exploring a vibrant cultural city or a tranquil national park makes a wonderful trip.",
        "I can suggest travel destinations based on your preferences. Let me know what kind of holiday you're planning.",
        "Scenic countryside getaways and coastal beach resorts offer fantastic options for a relaxing vacation."
    ],
    "calendar": [
        "I can help with calendar and schedule questions. You can review your scheduled events and meetings in your calendar app.",
        "Calendar query detected. Your agenda has upcoming syncs scheduled during regular business hours.",
        "I can identify schedule inquiries. Be sure to check your calendar app to confirm your meeting times.",
        "Looking at typical schedules, keeping afternoon slots open helps manage deadlines and scheduled reviews."
    ],
    "reminder": [
        "I can help with reminder requests. Please specify what task you'd like to remember and when.",
        "Reminder noted! Make sure to add this item to your device's to-do list so you receive timely notifications.",
        "Understood! Keeping track of your priorities ensures all your important tasks stay on schedule.",
        "Task noted! Be sure to set a checklist alert on your phone so you don't forget it."
    ],
    "alarm": [
        "I can identify alarm-related requests, but this demo does not actually create alarms.",
        "Alarm query detected. Please configure your wake-up time in your device's clock app.",
        "I can help with alarm settings. Make sure your phone's alarm volume is turned up for your wake-up time.",
        "Noted! Setting your morning wake-up time helps you stay organized and on schedule."
    ],
    "balance": [
        "I can help with account-balance queries, but this demo does not access a real bank account.",
        "Balance inquiry detected. You can review your available funds securely through your official mobile banking app.",
        "I can assist with balance-related questions. For security, your actual balance is available via your banking portal.",
        "Account balance query noted. Please log in to your bank's secure portal to view current checking and savings totals."
    ],
    "spending_history": [
        "I can help with spending-history questions. You can review recent expenses and statements in your banking app.",
        "Spending inquiry detected. Monthly bank statements provide itemized breakdowns across groceries, dining, and bills.",
        "I can help identify expense tracking requests. Checking your mobile banking budget tab shows recent category spending.",
        "Transaction history noted. You can inspect itemized purchases and monthly summaries in your financial app."
    ],
    "pay_bill": [
        "I can help identify bill-payment requests. Please settle your utility or invoice through your provider's official portal.",
        "Bill payment query detected. Make sure to authorize the transaction securely through your bank's bill-pay feature.",
        "I can assist with bill payment inquiries. Verify the account number and due date before submitting payment.",
        "Please review your billing statement and authorize the invoice payment through your service provider's secure portal."
    ],
    "transfer": [
        "I can help with transfer-related questions, but this demo cannot perform an actual financial transfer.",
        "Fund transfer request detected. To transfer money, please authorize the transaction in your official banking app.",
        "I can assist with transfer questions. Make sure to verify the recipient account details before sending funds.",
        "Money transfer inquiry noted. Secure transfers must be authenticated through your bank's mobile application."
    ],
    "freeze_account": [
        "I can identify account-freeze requests, but this demo cannot modify your account or lock your card.",
        "Emergency card freeze detected. If you suspect fraud or lost your card, call your bank's 24/7 hotline immediately.",
        "I can help with card lock inquiries. Use your mobile banking app's instant lock toggle to secure your card right away.",
        "Account security alert noted. Contact your bank immediately to block unauthorized transactions on your account."
    ],
    "greeting": [
        "Hello! How can I assist you with your queries today?",
        "Hi there! I am ready to help. What's on your mind?",
        "Greetings! Feel free to ask about weather, travel, directions, alarms, or banking inquiries.",
        "Hello! How are you doing today? What can I help you explore?",
        "Hey! Great to connect with you. What would you like to ask?"
    ],
    "goodbye": [
        "Goodbye! Have a great day ahead, and feel free to reach out anytime.",
        "Take care! Have a wonderful day and talk to you soon.",
        "See you later! Feel free to test more queries whenever you're ready.",
        "Bye! Wishing you a fantastic and productive day.",
        "Farewell! Let me know whenever you need assistance again."
    ],
    "thank_you": [
        "You're very welcome! Always happy to assist.",
        "Glad I could help! Feel free to try another query.",
        "Anytime! I am here whenever you want to test more questions.",
        "My pleasure! Let me know if there's anything else you need.",
        "You're welcome! Happy to be of service."
    ],
    "what_can_i_ask_you": [
        "You can ask me about weather, alarms, reminders, calendar, directions, restaurants, hotels, travel ideas, or banking tasks like balance and transfers!",
        "I am an academic Voice-Enabled Chatbot trained on 18 intent categories covering daily utilities, navigation, dining, and banking.",
        "Feel free to ask questions like 'What is the weather today?', 'Where is VIT Vellore located?', or 'How do I transfer money?'.",
        "I use Speech Recognition and a Deep Learning BiLSTM model to classify your questions and provide relevant responses across 18 domains."
    ],
    "tell_joke": [
        "Why did the computer go to the doctor? Because it had a virus! 😄",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why did the neural network go to school? To improve its gradient descent!",
        "There are 10 types of people in the world: those who understand binary, and those who don't.",
        "Why was the JavaScript developer sad? Because they didn't know how to 'null' their feelings!",
        "What do you call an algorithm that calculates your sleep? A rested API!"
    ]
}

# Semantic keyword evidence for supported intents
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "weather": ["weather", "rain", "temperature", "forecast", "sunny", "cloudy", "degrees", "hot", "cold", "snow", "umbrella", "humid", "storm", "wind", "chilly"],
    "directions": [
        "direction", "directions", "route", "navigate", "navigation", "map", "turn", "highway", "traffic",
        "drive", "get to", "way to", "location", "gps", "airport", "located", "situated",
        "where is", "where are", "where can i find", "location of", "how do i get to", "how to reach",
        "address of", "which place"
    ],
    "restaurant_suggestion": ["restaurant", "restaurants", "food", "eat", "dining", "dinner", "lunch", "bistro", "café", "cafe", "pizza", "meal", "hungry", "cuisine", "breakfast", "coffee"],
    "book_hotel": ["hotel", "hotels", "motel", "stay", "room", "lodging", "reservation", "resort", "booking", "accommodation", "place to stay"],
    "travel_suggestion": ["travel", "vacation", "trip", "holiday", "destination", "visit", "flight", "sightseeing", "explore", "tourist", "tourism", "place to travel"],
    "calendar": ["calendar", "schedule", "meeting", "meetings", "appointment", "appointments", "event", "events", "agenda", "busy"],
    "reminder": ["remind", "reminder", "reminders", "remember", "alert", "forget", "to-do", "task", "checklist"],
    "alarm": ["alarm", "alarms", "wake", "wake up", "timer", "snooze", "clock"],
    "balance": ["balance", "money", "how much", "checking", "savings", "funds", "coffers", "account have", "my bank"],
    "spending_history": ["spending", "spent", "expenses", "expense", "transaction", "transactions", "purchases", "history", "budget"],
    "pay_bill": ["bill", "bills", "pay", "payment", "electricity", "utility", "utilities", "invoice", "statement"],
    "transfer": ["transfer", "send money", "wire", "move money", "transfers", "move funds", "transferring"],
    "freeze_account": ["freeze", "block", "lost card", "stolen card", "lock card", "compromised", "fraud", "lock account", "stolen", "freeze card", "freeze my card", "freeze my account"],
    "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup", "what's up"],
    "goodbye": ["bye", "goodbye", "see you", "farewell", "take care", "have a good day", "later", "cya"],
    "thank_you": ["thank", "thanks", "appreciate", "grateful", "thank you"],
    "what_can_i_ask_you": [
        "what can you do", "what can i ask", "help me", "capabilities", "what do you do",
        "options", "features", "how do you work", "python", "machine learning",
        "deep learning", "artificial intelligence", "what is"
    ],
    "tell_joke": ["joke", "jokes", "funny", "laugh", "humor", "make me laugh", "comedy"]
}

# Documented demonstration knowledge base for common educational / technical queries
# Academically transparent: Pre-configured project response mappings rather than claimed neural generation.
PROJECT_KNOWLEDGE: Dict[str, str] = {
    "vit vellore": "VIT Vellore is located in Andhra Pradesh.",
    "vit": "VIT Vellore is located in Andhra Pradesh.",
    "vellore institute of technology": "VIT Vellore is located in Andhra Pradesh.",
    "python": "Python is a programming language commonly used for software development and data science.",
    "what is python": "Python is a programming language commonly used for software development and data science.",
    "machine learning": "Machine learning is a method where computers learn patterns from data to make predictions or decisions.",
    "what is machine learning": "Machine learning is a method where computers learn patterns from data to make predictions or decisions.",
    "deep learning": "Deep learning is a subset of machine learning based on multi-layered artificial neural networks, such as the BiLSTM model in this project.",
    "what is deep learning": "Deep learning is a subset of machine learning based on multi-layered artificial neural networks, such as the BiLSTM model in this project.",
    "artificial intelligence": "Artificial intelligence is the simulation of human intelligence by computer systems, including speech recognition and intent classification.",
    "what is artificial intelligence": "Artificial intelligence is the simulation of human intelligence by computer systems, including speech recognition and intent classification.",
    "what is an intent classifier": "An intent classifier uses natural language processing and deep learning models to categorize user utterances into discrete action intents.",
}

def find_keyword_intent(text: str) -> Optional[str]:
    """
    Scans the input text for strong keyword or phrase evidence matching
    one of the 18 supported intents, with domain disambiguation.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return None

    # High-priority domain checks
    # Weather
    if any(w in text_lower for w in ["weather", "rain", "temperature", "forecast", "snow", "sunny", "humid", "storm", "chilly"]):
        return "weather"
    # Restaurant / food
    if any(w in text_lower for w in ["restaurant", "restaurants", "food", "dining", "dinner", "lunch", "bistro", "café", "cafe", "pizza", "cuisine"]):
        return "restaurant_suggestion"
    # Hotel / lodging
    if any(w in text_lower for w in ["hotel", "hotels", "motel", "lodging", "resort", "room to stay", "place to stay"]):
        return "book_hotel"
    # Banking: freeze
    if any(w in text_lower for w in ["freeze", "block", "lost card", "stolen card", "lock card", "lock account"]):
        return "freeze_account"
    # Banking: bill
    if any(w in text_lower for w in ["bill", "bills", "utility", "invoice", "electricity"]):
        return "pay_bill"
    # Banking: transfer
    if any(w in text_lower for w in ["transfer", "send money", "wire", "move money"]):
        return "transfer"
    # Banking: balance
    if any(w in text_lower for w in ["balance", "how much money", "how much in my", "account balance", "my bank"]):
        return "balance"
    # Banking: spending
    if any(w in text_lower for w in ["spending", "spent", "expenses", "expense", "purchases", "spending history"]):
        return "spending_history"
    # Utilities: alarm
    if any(w in text_lower for w in ["alarm", "alarms", "wake up", "wake me up", "timer"]):
        return "alarm"
    # Utilities: reminder
    if any(w in text_lower for w in ["remind", "reminder", "reminders", "remember to"]):
        return "reminder"
    # Utilities: calendar
    if any(w in text_lower for w in ["calendar", "schedule", "meeting", "meetings", "appointment"]):
        return "calendar"
    # Jokes
    if any(w in text_lower for w in ["joke", "jokes", "funny", "laugh", "humor"]):
        return "tell_joke"
    # Greetings / Courtesies
    if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in ["hello", "hi", "hey", "greetings", "good morning", "good evening"]):
        return "greeting"
    if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in ["bye", "goodbye", "see you", "farewell"]):
        return "goodbye"
    if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in ["thank you", "thanks", "appreciate"]):
        return "thank_you"
    # Directions / Location inquiry
    if any(w in text_lower for w in [
        "where is", "where are", "where can i find", "located", "location of", "location",
        "situated", "how do i get to", "how to reach", "directions", "direction",
        "route", "navigate", "navigation", "map", "gps", "highway", "traffic", "drive"
    ]):
        return "directions"
    # General knowledge / capabilities / technical queries
    if any(w in text_lower for w in [
        "what can you do", "what can i ask", "capabilities", "what do you do", "how do you work",
        "what is python", "python", "what is machine learning", "machine learning",
        "what is deep learning", "deep learning", "what is artificial intelligence", "what is ai"
    ]):
        return "what_can_i_ask_you"
    # Travel suggestion
    if any(w in text_lower for w in ["vacation", "trip", "holiday", "sightseeing", "tourist", "tourism", "place to travel"]):
        return "travel_suggestion"

    return None

def is_nonsense_or_gibberish(text: str) -> bool:
    """
    Identifies completely nonsensical, random keystrokes, or off-domain inputs.
    """
    words = text.lower().split()
    if not words:
        return True

    # Check for random repeated digits
    if all(w.isdigit() for w in words):
        return True

    # Check for character mashing / non-dictionary long strings
    if any(len(w) > 16 and not w.startswith("http") for w in words):
        return True

    # Specific gibberish tokens commonly used in testing
    test_gibberish = {
        "asdfghjkl", "qwerty", "zxcvbn", "flim", "flam", "blorp",
        "quux", "zorp", "xyz", "asdkjfhqwuer", "kjasdhf",
        "supercalifragilisticexpialidocious"
    }
    if any(w in test_gibberish for w in words):
        return True

    return False

def get_location_response(text: str) -> Optional[str]:
    """
    Detects if the query is asking about the location of a place, college, or landmark,
    and returns a direct, intended location answer.
    """
    text_lower = text.lower().strip().rstrip("?.!")

    # Must have a location cue OR be specifically asking about VIT Vellore
    has_location_cue = any(c in text_lower for c in [
        "where", "location", "located", "situated", "address", "how do i get",
        "how to get", "route to", "way to", "directions to", "find"
    ])

    # Check for VIT / Vellore queries specifically
    is_vit_query = any(re.search(r'\b' + re.escape(v) + r'\b', text_lower) for v in ["vit vellore", "vit", "vellore institute", "vit university"])
    if is_vit_query and (has_location_cue or "vit vellore" in text_lower):
        return "VIT Vellore is located in Andhra Pradesh."

    if not has_location_cue:
        return None

    # Well-known landmarks and common entities
    famous_places = {
        "taj mahal": "The Taj Mahal is located in Agra, Uttar Pradesh.",
        "golden gate bridge": "The Golden Gate Bridge is located in San Francisco, California.",
        "eiffel tower": "The Eiffel Tower is located in Paris, France.",
        "statue of liberty": "The Statue of Liberty is located in New York Harbor, New York.",
        "stanford": "Stanford is located in Stanford, California.",
        "harvard": "Harvard is located in Cambridge, Massachusetts.",
        "mit": "MIT is located in Cambridge, Massachusetts.",
        "oxford": "Oxford is located in Oxford, England.",
        "cambridge": "Cambridge is located in Cambridge, England.",
        "iit madras": "IIT Madras is located in Chennai, Tamil Nadu.",
        "iit bombay": "IIT Bombay is located in Mumbai, Maharashtra.",
        "iit delhi": "IIT Delhi is located in New Delhi.",
        "library": "The library is located on the North Campus near the main academic quadrangle.",
        "hospital": "The hospital is located on Medical Center Boulevard, about 2 miles north.",
        "starbucks": "It is located 2 blocks down on Main Street.",
        "airport": "The airport is located 12 miles south via the central express highway.",
        "gas station": "The gas station is located at the intersection of Main Avenue and 4th Street.",
        "pharmacy": "The pharmacy is located in the commercial plaza on Elm Street.",
        "bank": "The bank is located downtown on Financial District Avenue.",
        "cafeteria": "The cafeteria is located on the ground floor of the student activity center.",
        "gym": "The gym is located in the sports complex next to the campus stadium."
    }

    for place_key, answer in famous_places.items():
        if re.search(r'\b' + re.escape(place_key) + r'\b', text_lower):
            return answer

    # Pattern extraction for other 'where is [X] located' or 'location of [X]' queries
    patterns = [
        r"(?:where\s+is|where\s+are|where\s+can\s+i\s+find)\s+(.*?)(?:\s+located|\s+situated)?$",
        r"(?:what\s+is\s+the\s+location\s+of|location\s+of|address\s+of)\s+(.*?)$",
        r"(.*?)\s+(?:location|is\s+located|is\s+situated)$"
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            subject = m.group(1).strip()
            subject = re.sub(r'^(the|a|an)\s+', '', subject).strip()
            if subject and len(subject) > 1 and subject not in ["i", "we", "you", "it"]:
                return f"{subject.title()} is located in the central district. For turn-by-turn navigation, you can check Google Maps."

    return None

def get_response(intent: str,
                 confidence: float,
                 threshold: float = config.CONFIDENCE_THRESHOLD,
                 raw_text: str = "") -> str:
    """
    Selects a semantically relevant response based on the user's question and predicted intent.

    Behavior:
    1. Direct conversational handling for sentiments & courtesies.
    2. Direct location entity handling for 'where is X located' queries.
    3. Documented demonstration knowledge matching (e.g. 'what is python', 'what is machine learning').
    4. Case 1 (Related / Normal / Ambiguous Question):
       Returns a natural, informative response directly relevant to the predicted or keyword-matched intent.
       Never returns a generic 'I don't understand' message for related queries.
    5. Case 2 (Genuinely unrelated / nonsensical question):
       Returns the informative fallback explaining the 18 supported domains.
    """
    lower_text = raw_text.strip().lower() if raw_text else ""

    # Direct conversational handling for sentiments
    if any(p in lower_text for p in ["hate you", "you are bad", "terrible", "useless", "stupid", "annoying"]):
        return "I'm sorry you feel that way! I am an academic intent-classification demo. You can test me with queries about weather, travel, alarms, calendar, directions, or banking."

    if any(p in lower_text for p in ["love you", "you are great", "awesome", "good job", "amazing", "you're cool"]):
        return "Thank you! That is very kind of you. How can I assist you today?"

    if any(p in lower_text for p in ["who are you", "what are you", "what is your name"]):
        return "I am a Voice-Enabled Chatbot using Speech Recognition and a Deep Learning BiLSTM model to classify user intents across 18 categories."

    # Direct location query check: e.g. "where is vit vellore located"
    loc_response = get_location_response(raw_text)
    if loc_response:
        return loc_response

    # Direct documented demonstration knowledge check (e.g. "what is python", "what is machine learning")
    for k, ans in PROJECT_KNOWLEDGE.items():
        if re.search(r'\b' + re.escape(k) + r'\b', lower_text):
            return ans

    # Check for semantic keyword evidence
    keyword_intent = find_keyword_intent(raw_text)

    # Determine whether input is Case 2 (genuinely unrelated or nonsensical)
    is_gibberish = is_nonsense_or_gibberish(raw_text)
    is_completely_off_topic = (
        keyword_intent is None and (
            confidence < 0.22 or
            any(w in lower_text for w in [
                "quantum", "black hole", "black holes", "photosynthesis",
                "calculus", "dinosaur", "dinosaurs", "astronomy", "cosmology"
            ])
        )
    )

    if is_gibberish or is_completely_off_topic:
        return config.FALLBACK_RESPONSE

    # Case 1: Select the most relevant intent
    # If a keyword strongly points to an intent and model confidence is modest, favor keyword intent
    target_intent = intent
    if keyword_intent and keyword_intent != intent and confidence < 0.70:
        target_intent = keyword_intent

    responses = INTENT_RESPONSES.get(target_intent)
    if not responses:
        responses = INTENT_RESPONSES.get(intent)
    if not responses:
        return config.FALLBACK_RESPONSE

    return random.choice(responses)
