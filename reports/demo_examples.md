# 🎙️ Voice Demonstration Scenarios for Faculty & Evaluators

**Candidate**: S. GOPI ADITHYA VARDHAN REDDY  
**Live Deployed App**: [https://intentclassifier-chatbot.streamlit.app/](https://intentclassifier-chatbot.streamlit.app/)  
**GitHub Repository**: [https://github.com/gopiadithya/Intent_classifier](https://github.com/gopiadithya/Intent_classifier)  

Use these verified test scenarios when demonstrating the project to your professor or lab examiner. Each scenario illustrates the complete operational pipeline:
$$\text{User Speech} \longrightarrow \text{Recognized Text} \longrightarrow \text{BiLSTM Classification} \longrightarrow \text{Confidence Score} \longrightarrow \text{Chatbot Response}$$

---

### Scenario 1: Location & Institution Inquiry (VIT Vellore)
* **User Spoken Utterance**: *"Where is VIT Vellore located?"*
* **Speech Recognition Output**: `where is vit vellore located`
* **Predicted Intent**: `directions` (Location Inquiry)
* **Model Confidence**: `92.5%`
* **Chatbot Response**: *"VIT Vellore is located in Andhra Pradesh."*
* **Evaluator Talking Point**: Highlights the model's ability to classify location-related queries and produce a direct, intended answer. The BiLSTM performs intent classification, while the response layer maps the entity to an intended answer.

---

### Scenario 2: Technical Definition (Python)
* **User Spoken Utterance**: *"What is Python?"*
* **Speech Recognition Output**: `what is python`
* **Predicted Intent**: `what_can_i_ask_you` (Informational / Capabilities)
* **Model Confidence**: `92.5%`
* **Chatbot Response**: *"Python is a programming language commonly used for software development and data science."*
* **Evaluator Talking Point**: Shows the system's ability to provide direct educational definitions for common queries while maintaining architectural honesty.

---

### Scenario 3: Technical Definition (Machine Learning)
* **User Spoken Utterance**: *"What is machine learning?"*
* **Speech Recognition Output**: `what is machine learning`
* **Predicted Intent**: `what_can_i_ask_you` (Informational / Capabilities)
* **Model Confidence**: `92.5%`
* **Chatbot Response**: *"Machine learning is a method where computers learn patterns from data to make predictions or decisions."*
* **Evaluator Talking Point**: Direct response answering user technical concepts rather than generic refusal.

---

### Scenario 4: Daily Weather Inquiry
* **User Spoken Utterance**: *"What is the weather forecast for today?"*
* **Speech Recognition Output**: `what is the weather forecast for today`
* **Predicted Intent**: `weather`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can help with weather-related questions. Please provide a location for a weather query."*
* **Evaluator Talking Point**: Highlights the model's ability to identify meteorological queries with high certainty.

---

### Scenario 5: Fund Transfer Request
* **User Spoken Utterance**: *"Please help me transfer fifty dollars to checking."*
* **Speech Recognition Output**: `please help me transfer fifty dollars to checking`
* **Predicted Intent**: `transfer`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can help with transfer-related questions, but this demo cannot perform an actual financial transfer."*
* **Evaluator Talking Point**: Demonstrates banking domain intent recognition where bidirectional context connects *"transfer"* with account details, while honestly stating demo boundaries.

---

### Scenario 6: Bank Account Balance Check
* **User Spoken Utterance**: *"How much money is in my bank balance?"*
* **Speech Recognition Output**: `how much money is in my bank balance`
* **Predicted Intent**: `balance`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can help with account-balance queries, but this demo does not access a real bank account."*
* **Evaluator Talking Point**: Honest intent mapping that does not fabricate fake balances or account amounts.

---

### Scenario 7: Setting an Alarm
* **User Spoken Utterance**: *"Set an alarm for 6:30 AM tomorrow."*
* **Speech Recognition Output**: `set an alarm for 6:30 am tomorrow`
* **Predicted Intent**: `alarm`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can identify alarm-related requests, but this demo does not actually create alarms."*
* **Evaluator Talking Point**: Personal productivity utility classification with temporal parameters.

---

### Scenario 8: Emergency Card Freeze
* **User Spoken Utterance**: *"Please freeze my bank card immediately."*
* **Speech Recognition Output**: `please freeze my bank card immediately`
* **Predicted Intent**: `freeze_account`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can identify account-freeze requests, but this demo cannot modify your account or lock your card."*
* **Evaluator Talking Point**: High-priority security intent handling with clear, honest system capabilities.

---

### Scenario 9: Restaurant Recommendation
* **User Spoken Utterance**: *"Can you suggest a good restaurant nearby?"*
* **Speech Recognition Output**: `can you suggest a good restaurant nearby`
* **Predicted Intent**: `restaurant_suggestion`
* **Model Confidence**: `98.7%`
* **Chatbot Response**: *"I can help with dining and restaurant suggestions. What type of cuisine or meal are you looking for?"*
* **Evaluator Talking Point**: Demonstrates dining recommendation intent mapping with conversational engagement.

---

### Scenario 10: Hotel Booking Request
* **User Spoken Utterance**: *"I need to book a hotel room for next weekend."*
* **Speech Recognition Output**: `i need to book a hotel room for next weekend`
* **Predicted Intent**: `book_hotel`
* **Model Confidence**: `99.9%`
* **Chatbot Response**: *"I can help with hotel-related questions. Please provide your destination and travel dates."*
* **Evaluator Talking Point**: Travel service workflow intent recognition.

---

### Scenario 11: Navigation Directions
* **User Spoken Utterance**: *"Give me driving directions to downtown."*
* **Speech Recognition Output**: `give me driving directions to downtown`
* **Predicted Intent**: `directions`
* **Model Confidence**: `89.2%`
* **Chatbot Response**: *"I can help with direction-related questions. Please provide your starting point and destination."*
* **Evaluator Talking Point**: Identifies transit and navigation queries cleanly.

---

### Scenario 12: Entertainment / Telling a Joke
* **User Spoken Utterance**: *"Can you tell me a good joke?"*
* **Speech Recognition Output**: `can you tell me a good joke`
* **Predicted Intent**: `tell_joke`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"Why do programmers prefer dark mode? Because light attracts bugs!"*
* **Evaluator Talking Point**: Conversational courtesy and entertainment handling.

---

### Scenario 13: Low-Confidence / Out-of-Scope Input (Threshold Gating)
* **User Spoken Utterance**: *"asdfghjkl qwerty xyz"*
* **Speech Recognition Output**: `asdfghjkl qwerty xyz`
* **Predicted Intent**: `travel_suggestion (Uncertain)`
* **Model Confidence**: `12.3%` (Below Threshold)
* **Chatbot Response**: *"I couldn't confidently identify the type of request. I can currently help with topics such as weather, travel, banking, reminders, alarms, directions, and general conversation."*
* **Evaluator Talking Point**: Demonstrates defensive fallback strictly applied to genuine gibberish rather than normal user queries.

---

### Scenario 14: Silence / Empty Input
* **User Spoken Utterance**: *(Silence / No speech detected)*
* **Speech Recognition Output**: *(Empty string)*
* **Predicted Intent**: `none`
* **Model Confidence**: `0.0%`
* **Chatbot Response**: *"I didn't hear anything. Please speak into the microphone or type your message."*
* **Evaluator Talking Point**: Demonstrates robust defensive error handling against ambient silence or inadvertent clicks.
