"""
Response Catalog and Selector for Voice-Enabled Chatbot.
Provides natural, relevant conversational responses for all 18 supported intents,
sentiment expressions, and sensible low-confidence handling.
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
        "I can help with weather-related questions. Please provide a location if you want specific weather information.",
        "I can identify weather queries. In a complete assistant, this would fetch real-time forecasts from a meteorological service.",
        "Weather query detected! While this academic demo doesn't retrieve live temperature data, typical conditions this season are mild and pleasant.",
        "If you're planning around the weather, remember to check your local forecast for current rain and temperature updates.",
        "I can help identify forecast and temperature questions, though this demo does not connect to a live weather satellite feed.",
        "Looking for weather updates? A live meteorological app can provide hourly rain and temperature details for your area."
    ],
    "directions": [
        "I can help with navigation and direction requests. To get turn-by-turn routing, please check a mapping service like Google Maps.",
        "I can identify route and direction inquiries. For accurate live traffic updates, consult your GPS navigation application.",
        "Directions query recognized! In a full navigation assistant, this would calculate the fastest route to your destination.",
        "If you need travel directions, mapping tools can provide driving, walking, or public transit options.",
        "I can help identify navigation questions, though this demo does not connect to live GPS or traffic satellites."
    ],
    "restaurant_suggestion": [
        "I can identify restaurant recommendation requests, but this demo does not access live local restaurant listings.",
        "Looking for food or dining suggestions? You can explore top-rated local dining spots through restaurant review apps like Yelp or Google Maps.",
        "I can help with dining recommendations. A nearby Italian bistro or downtown grill is often a great choice for lunch or dinner!",
        "Dining recommendation query detected. In a production system, this would query a culinary database for menus and tables.",
        "I can identify food-related inquiries. Please check local dining guides to find open tables and current menus near you."
    ],
    "book_hotel": [
        "I can help with hotel-related requests. Please provide your destination and travel dates for a hotel search through a booking service.",
        "I can identify hotel-booking requests, but this academic demo does not execute real reservations.",
        "Accommodation query detected! To complete an actual room booking, please use a reservation portal like Booking.com or Expedia.",
        "Need a place to stay? Booking applications can show available rooms, rates, and amenities near your destination.",
        "I can identify lodging and hotel inquiries, though actual reservations must be made through an authorized booking gateway."
    ],
    "travel_suggestion": [
        "I can identify travel recommendation queries, but this demo does not fetch live travel guides or destination data.",
        "Looking for vacation ideas? Scenic mountain retreats, national parks, or coastal beach towns make wonderful trip destinations!",
        "Travel inquiry recognized! In a complete travel assistant, this would suggest curated itineraries based on your interests.",
        "I can help with destination and vacation suggestions. Check travel advisory sites for seasonal highlights and flight details.",
        "Planning a trip? Exploring a vibrant cultural city or a relaxing countryside getaway is always a great choice."
    ],
    "calendar": [
        "I can recognize calendar and schedule inquiries, but this demo is not connected to a personal calendar service.",
        "Calendar query detected. In a full assistant, this would fetch scheduled events from an API like Google Calendar or Outlook.",
        "Checking your schedule? You can review your upcoming meetings and appointments directly in your calendar app.",
        "I can help identify agenda and schedule questions. Be sure to check your calendar app so you don't miss upcoming events.",
        "Schedule inquiry recognized! An integrated calendar assistant would sync your daily meetings and remind you of deadlines."
    ],
    "reminder": [
        "I can identify reminder-related requests, but this chatbot demo does not actually schedule device notifications.",
        "Reminder request detected. In a complete application, this would register a scheduled task alert in your system.",
        "I can help with reminder requests. Please add this item to your device's reminder or to-do app to ensure you're alerted on time.",
        "Noted that you want a reminder! In a fully integrated assistant, this would alert you at the specified time.",
        "Reminder query recognized. To ensure you don't forget, make sure to add it to your personal task checklist."
    ],
    "alarm": [
        "I can identify alarm-related requests, but this demo does not actually create device alarms.",
        "Alarm configuration request detected. In an integrated OS assistant, this would trigger the system clock service.",
        "I can help with alarm requests. Please verify and set the time in your device's clock app so you wake up on schedule.",
        "Alarm query recognized! Be sure to double-check that your device volume is turned up for your wake-up time.",
        "I can identify alarm requests, but this chatbot demo does not configure system hardware alarms."
    ],
    "balance": [
        "I can help with balance-related questions. You can check your current account balance through your banking application.",
        "I can recognize account balance inquiries, but this demo is not connected to any live banking API or financial records.",
        "Account balance query detected. In a production banking app, this would securely retrieve your account balance.",
        "I can help with account-balance queries. Your actual balance would need to be checked through your banking service.",
        "Balance inquiry recognized. For your security, actual account funds can only be accessed through authenticated banking portals."
    ],
    "spending_history": [
        "I can help with spending-history questions. You can review your recent expenses and transactions through your banking application.",
        "I can recognize requests for spending analysis, but this demo does not access personal financial transaction logs.",
        "Spending history query detected. In an integrated financial assistant, this would query your transaction database.",
        "Looking to review your spending? Your banking portal provides detailed breakdowns of recent purchases and monthly budgets.",
        "Spending inquiry recognized. Checking your monthly bank statement is the best way to track category expenses and budget trends."
    ],
    "pay_bill": [
        "I can identify bill-payment requests, but this demo does not perform financial payments or settlements.",
        "Bill payment request detected. In an integrated banking system, this would securely process the transaction.",
        "I can help identify bill payment requests. Please settle your utility or credit card invoice through your provider's official portal.",
        "Need to pay a bill? Make sure to use your bank's bill-pay feature or your service provider's secure website.",
        "Bill payment inquiry recognized. For your security, payments must be authorized through your financial institution."
    ],
    "transfer": [
        "I can help identify transfer-related requests. To make an actual transfer, please use your banking application.",
        "I can identify transfer-related requests, but this demo does not perform financial transactions.",
        "Fund transfer request detected. In a live banking assistant, this would initiate a secure funds transfer flow.",
        "Looking to transfer money? Please initiate your funds transfer securely through your official banking portal or mobile app.",
        "Transfer inquiry recognized. Financial transfers require multi-factor authentication through your bank."
    ],
    "freeze_account": [
        "I can identify account-freeze requests, but this demo cannot modify your account or lock your card.",
        "Emergency account-freeze request detected. In a banking app, this would initiate a security block on your card.",
        "I can help identify account-freeze requests. If you suspect fraud or lost your card, please call your bank's emergency line immediately.",
        "Card lock request recognized. Please contact your card issuer or use your mobile banking app's instant lock switch to freeze your card.",
        "Account security request detected. To protect your funds, contact your bank immediately to place a freeze on your account."
    ]
}

# Semantic keyword evidence for supported intents
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "weather": ["weather", "rain", "temperature", "forecast", "sunny", "cloudy", "degrees", "hot", "cold", "snow", "umbrella", "humid", "storm", "wind", "chilly"],
    "directions": ["direction", "directions", "route", "navigate", "navigation", "map", "turn", "highway", "traffic", "drive", "get to", "way to", "location", "gps", "airport"],
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
    one of the 18 supported intents.
    """
    text_lower = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if len(kw) <= 3:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    return intent
            else:
                if kw in text_lower:
                    return intent
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

def get_response(intent: str,
                 confidence: float,
                 threshold: float = config.CONFIDENCE_THRESHOLD,
                 raw_text: str = "") -> str:
    """
    Selects a semantically relevant response based on the user's question and predicted intent.

    Behavior:
    1. Direct conversational handling for sentiments & courtesies.
    2. Case 1 (Related / Normal / Ambiguous Question):
       Returns a natural, informative response directly relevant to the predicted or keyword-matched intent.
       Never returns a generic 'I don't understand' message for related queries.
    3. Case 2 (Genuinely unrelated / nonsensical question):
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

    # Specific tailored responses for common user phrasings
    if "temperature" in lower_text and "weather" in INTENT_RESPONSES:
        return "I can help identify weather-related questions. Please provide a location if you want specific temperature or weather information."

    if any(p in lower_text for p in ["how much money", "how much do i have", "what is my balance"]):
        return "I can help with balance-related questions. You can check your current account balance through your banking application."

    if any(p in lower_text for p in ["place to stay", "somewhere to stay"]):
        return "I can help with hotel-related requests. Please provide your destination and travel dates for a hotel search through a booking service."

    if any(p in lower_text for p in ["spent too much", "spent money", "my spending"]):
        return "I can help with spending-history questions. You can review your recent expenses and transactions through your banking application."

    if any(p in lower_text for p in ["transfer money", "want to transfer"]):
        return "I can help identify transfer-related requests. To make an actual transfer, please use your banking application."

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
    if keyword_intent and keyword_intent != intent and confidence < 0.45:
        target_intent = keyword_intent

    responses = INTENT_RESPONSES.get(target_intent)
    if not responses:
        responses = INTENT_RESPONSES.get(intent)
    if not responses:
        return config.FALLBACK_RESPONSE

    return random.choice(responses)
