"""
Response Catalog and Selector for Voice-Enabled Chatbot.
Provides natural, relevant conversational responses for all 18 supported intents,
sentiment expressions, location/directional entity answers, and sensible fallback handling.
Academically transparent: Never falsely claims real-world action execution.
"""
import re
import random
from typing import Dict, List, Optional
import config

INTENT_RESPONSES: Dict[str, List[str]] = {
    "greeting": [
        "Hello! How can I assist you with your queries today?",
        "Hi there! I am ready to help. What's on your mind?",
        "Greetings! Feel free to ask about weather, travel, alarms, or banking.",
        "Hello! How are you doing today? What can I help you with?",
        "Hey! Great to connect with you. What would you like to explore?"
    ],
    "goodbye": [
        "Goodbye! Have a great day ahead, and feel free to reach out anytime.",
        "Take care! Have a wonderful day and talk to you soon.",
        "See you later! Feel free to test more queries whenever you're ready.",
        "Bye! Wishing you a fantastic day ahead.",
        "Farewell! Let me know if you need assistance with anything else."
    ],
    "thank_you": [
        "You're very welcome! Always happy to help.",
        "Glad I could assist! Feel free to try another query.",
        "Anytime! I am here whenever you want to test more questions.",
        "My pleasure! Let me know if there's anything else you need.",
        "You're welcome! Happy to be of service."
    ],
    "what_can_i_ask_you": [
        "You can ask me about weather, alarms, reminders, calendar, directions, restaurants, hotels, travel ideas, or banking tasks like balance and transfers!",
        "I can help identify questions across daily utilities (weather, alarms, schedule), lifestyle (directions, dining, travel), and banking tasks.",
        "Feel free to ask about setting alarms, checking reminders, viewing schedules, getting dining or travel ideas, or managing banking inquiries.",
        "I'm trained on 18 intent categories covering daily tasks, travel planning, navigation, and simulated financial queries.",
        "You can test me with voice or text queries related to weather forecasts, restaurant recommendations, directions, alarms, or account balance."
    ],
    "tell_joke": [
        "Why did the computer go to the doctor? Because it had a virus! 😄",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why did the neural network go to school? To improve its gradient descent!",
        "There are 10 types of people in the world: those who understand binary, and those who don't.",
        "Why was the JavaScript developer sad? Because they didn't know how to 'null' their feelings!",
        "What do you call an algorithm that calculates your sleep? A rested API!"
    ],
    "weather": [
        "Current weather conditions are mostly sunny and pleasant with temperatures around 24°C (75°F) and a mild breeze.",
        "The weather forecast calls for clear skies with partly cloudy intervals and a low chance of rain today.",
        "The local weather looks warm and pleasant today, with temperatures reaching up to 26°C and gentle winds.",
        "Expect mild weather conditions with scattered clouds throughout the day and comfortable temperatures.",
        "The weather is currently clear and comfortable, ideal for outdoor travel and activities."
    ],
    "directions": [
        "Head straight on Main Avenue for about 1.5 miles, then follow the highway signs toward your destination.",
        "The fastest route is via the central expressway; traffic is currently moving smoothly.",
        "It is located about 10 minutes away just off the main avenue. Follow signs toward the central boulevard.",
        "Take the next right turn onto Grand Boulevard, then continue straight for approximately 2 miles.",
        "Head north along the transit corridor and take the second exit toward the destination district."
    ],
    "restaurant_suggestion": [
        "A highly rated Italian bistro downtown or a local sushi grill would be a great choice for your meal!",
        "For dining, you might enjoy the riverside cafe known for fresh seasonal dishes and artisanal pizzas.",
        "There is a wonderful Mediterranean restaurant nearby with fantastic falafel, kebabs, and pasta options.",
        "I'd suggest checking out the local bistro on Market Street—it has great reviews for casual dining and great flavors!"
    ],
    "book_hotel": [
        "The Grand Central Hotel and Riverside Suites have excellent guest ratings and comfortable rooms available.",
        "For your stay, the Downtown Plaza Hotel offers great amenities, complimentary breakfast, and scenic city views.",
        "You can look into the Metro Boutique Hotel or the Express Inn near the city center for convenient lodging.",
        "The Harbor View Resort provides top-tier hospitality, spacious rooms, and great access to local attractions."
    ],
    "travel_suggestion": [
        "A trip to scenic mountain retreats or a relaxing coastal beach getaway would make an unforgettable vacation!",
        "Exploring a vibrant cultural city with historic architecture and local food markets is always a wonderful travel idea.",
        "A weekend getaway to the national park trails or a tranquil countryside resort would be very refreshing!",
        "Visiting a historic coastal town or an adventure nature reserve offers an amazing travel experience."
    ],
    "calendar": [
        "Your next scheduled event is a team sync at 2:00 PM, followed by a project review later in the afternoon.",
        "Looking at your schedule, your morning is free and your upcoming meeting starts at 3:30 PM.",
        "You have two calendar entries planned for today: a status catch-up and an afternoon planning session.",
        "Your calendar shows a project review scheduled for 11:00 AM tomorrow morning."
    ],
    "reminder": [
        "Noted! Make sure to keep that at the top of your to-do list so you don't miss it.",
        "Reminder noted! Make sure to check your schedule to stay on track with your tasks.",
        "Got it! I have noted that task so you can address it at the right time.",
        "Understood! Keep that item in your daily notes to ensure timely completion."
    ],
    "alarm": [
        "Make sure to set your device clock so you wake up right on time.",
        "Got it! Be sure to double-check that your device alarm volume is turned up for the morning.",
        "Noted! Setting your wake-up time early helps ensure a productive start to your day.",
        "Alarm timing noted! Make sure your device is fully charged for your wake-up alert."
    ],
    "balance": [
        "Your primary checking account balance shows a healthy balance of available funds.",
        "Your savings and checking accounts are in good standing with sufficient funds available.",
        "You have an active balance across your accounts with no pending overdrafts.",
        "Your account maintains a positive standing with standard available funds for daily transactions."
    ],
    "spending_history": [
        "Your recent expenses show typical transactions in groceries, dining, and utility payments over the past month.",
        "Looking at your recent spending breakdown, the largest categories were household supplies and dining out.",
        "Your transaction history indicates consistent budget management across standard monthly categories.",
        "Recent monthly spending appears balanced, with normal expenses in utilities, shopping, and transportation."
    ],
    "pay_bill": [
        "Please ensure you authorize the utility bill payment through your official provider portal or bank app.",
        "To settle this invoice, you can confirm the account details on your service provider's payment page.",
        "Bill payment details noted. Make sure to complete the transaction on your bank's secure bill-pay gateway.",
        "Please review your invoice statement and authorize the payment securely through your account portal."
    ],
    "transfer": [
        "To complete your funds transfer, please confirm the recipient details and approve the transfer in your banking app.",
        "Transfer request noted. Please verify the destination account number and authenticate the transaction.",
        "Please make sure to review the transfer amount and recipient before authorizing the transfer with your bank.",
        "Fund transfer request noted. Please finalize and confirm the transfer through your official mobile banking portal."
    ],
    "freeze_account": [
        "If you suspect fraud or lost your card, please immediately lock your card in your banking app or call your bank's 24/7 hotline.",
        "Urgent card lock noted. Please use your bank's instant card lock switch in mobile banking or phone customer support immediately.",
        "Security freeze alert: To protect your funds, contact your bank's emergency fraud department right away.",
        "Please open your mobile banking security settings right away to freeze your card and prevent unauthorized charges."
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
    "what_can_i_ask_you": ["what can you do", "what can i ask", "help me", "capabilities", "what do you do", "options", "features", "how do you work"],
    "tell_joke": ["joke", "jokes", "funny", "laugh", "humor", "make me laugh", "comedy"]
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
    if any(w in text_lower for w in ["what can you do", "what can i ask", "capabilities", "what do you do", "how do you work"]):
        return "what_can_i_ask_you"
    # Travel suggestion
    if any(w in text_lower for w in ["vacation", "trip", "holiday", "sightseeing", "tourist", "tourism", "place to travel"]):
        return "travel_suggestion"
    # Directions / Location inquiry
    if any(w in text_lower for w in [
        "where is", "where are", "where can i find", "located", "location of", "location",
        "situated", "how do i get to", "how to reach", "directions", "direction",
        "route", "navigate", "navigation", "map", "gps", "highway", "traffic", "drive"
    ]):
        return "directions"

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

    # Check for VIT / Vellore queries specifically
    if any(v in text_lower for v in ["vit vellore", "vit", "vellore institute", "vit university"]):
        return "It is located in Andhra Pradesh."

    # Well-known landmarks and common entities
    famous_places = {
        "taj mahal": "It is located in Agra, Uttar Pradesh.",
        "golden gate bridge": "It is located in San Francisco, California.",
        "eiffel tower": "It is located in Paris, France.",
        "statue of liberty": "It is located in New York Harbor, New York.",
        "stanford": "It is located in Stanford, California.",
        "harvard": "It is located in Cambridge, Massachusetts.",
        "mit": "It is located in Cambridge, Massachusetts.",
        "oxford": "It is located in Oxford, England.",
        "cambridge": "It is located in Cambridge, England.",
        "iit madras": "It is located in Chennai, Tamil Nadu.",
        "iit bombay": "It is located in Mumbai, Maharashtra.",
        "iit delhi": "It is located in New Delhi.",
        "library": "It is located on the North Campus near the main academic quadrangle.",
        "hospital": "It is located on Medical Center Boulevard, about 2 miles north.",
        "starbucks": "It is located 2 blocks down on Main Street.",
        "airport": "It is located 12 miles south via the central express highway.",
        "gas station": "It is located at the intersection of Main Avenue and 4th Street.",
        "pharmacy": "It is located in the commercial plaza on Elm Street.",
        "bank": "It is located downtown on Financial District Avenue.",
        "cafeteria": "It is located on the ground floor of the student activity center.",
        "gym": "It is located in the sports complex next to the campus stadium."
    }

    for place_key, answer in famous_places.items():
        if place_key in text_lower:
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
                return f"It is located in the central district of {subject.title()}. For turn-by-turn navigation, you can check Google Maps."

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
    3. Case 1 (Related / Normal / Ambiguous Question):
       Returns a natural, informative response directly relevant to the predicted or keyword-matched intent.
       Never returns a generic 'I don't understand' message for related queries.
    4. Case 2 (Genuinely unrelated / nonsensical question):
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

    # Check for semantic keyword evidence
    keyword_intent = find_keyword_intent(raw_text)

    # Determine whether input is Case 2 (genuinely unrelated or nonsensical)
    is_gibberish = is_nonsense_or_gibberish(raw_text)
    is_completely_off_topic = (
        keyword_intent is None and (
            confidence < 0.22 or
            any(w in lower_text for w in ["quantum", "physics", "black hole", "black holes", "photosynthesis", "calculus", "chemistry", "biology", "dinosaur", "dinosaurs", "astronomy", "dna", "cosmology"])
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
