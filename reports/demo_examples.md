# 🎙️ Voice Demonstration Scenarios for Faculty & Evaluators

Use these verified test scenarios when demonstrating the project to your professor or lab examiner. Each scenario illustrates the complete operational pipeline:
$$\text{User Speech} \longrightarrow \text{Recognized Text} \longrightarrow \text{BiLSTM Classification} \longrightarrow \text{Confidence Score} \longrightarrow \text{Chatbot Response}$$

---

### Scenario 1: Daily Weather Inquiry
* **User Spoken Utterance**: *"What is the weather forecast for today?"*
* **Speech Recognition Output**: `what is the weather forecast for today`
* **Predicted Intent**: `weather`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"Weather query detected. The BiLSTM model classified your intent as 'weather'."*
* **Evaluator Talking Point**: Highlights the model's ability to identify meteorological queries with high certainty without fabricating external forecast data.

---

### Scenario 2: Fund Transfer Request
* **User Spoken Utterance**: *"Please help me transfer fifty dollars to checking."*
* **Speech Recognition Output**: `please help me transfer fifty dollars to checking`
* **Predicted Intent**: `transfer`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can identify transfer-related requests, but this demo does not perform financial transactions."*
* **Evaluator Talking Point**: Demonstrates banking domain intent recognition where bidirectional context connects *"transfer"* with account details, while honestly stating demo boundaries.

---

### Scenario 3: Bank Account Balance Check
* **User Spoken Utterance**: *"How much money is in my bank balance?"*
* **Speech Recognition Output**: `how much money is in my bank balance`
* **Predicted Intent**: `balance`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can recognize account balance inquiries, but this demo is not connected to any live banking API or financial records."*
* **Evaluator Talking Point**: Honest intent mapping that does not fabricate fake balances or account amounts.

---

### Scenario 4: Setting an Alarm
* **User Spoken Utterance**: *"Set an alarm for 6:30 AM tomorrow."*
* **Speech Recognition Output**: `set an alarm for 6:30 am tomorrow`
* **Predicted Intent**: `alarm`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can identify alarm-related requests, but this chatbot does not actually create device alarms."*
* **Evaluator Talking Point**: Personal productivity utility classification with temporal parameters.

---

### Scenario 5: Emergency Card Freeze
* **User Spoken Utterance**: *"Please freeze my bank card immediately."*
* **Speech Recognition Output**: `please freeze my bank card immediately`
* **Predicted Intent**: `freeze_account`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can identify account-freeze requests, but this demo cannot modify your account or lock your card."*
* **Evaluator Talking Point**: High-priority security intent handling with clear, honest system capabilities.

---

### Scenario 6: Restaurant Recommendation
* **User Spoken Utterance**: *"Can you suggest a good restaurant nearby?"*
* **Speech Recognition Output**: `can you suggest a good restaurant nearby`
* **Predicted Intent**: `restaurant_suggestion`
* **Model Confidence**: `99.9%`
* **Chatbot Response**: *"I can identify restaurant recommendation queries, but this demo does not access a live local business directory."*
* **Evaluator Talking Point**: Demonstrates dining recommendation intent mapping without inventing fake restaurants or ratings.

---

### Scenario 7: Hotel Booking Request
* **User Spoken Utterance**: *"I need to book a hotel room for next weekend."*
* **Speech Recognition Output**: `i need to book a hotel room for next weekend`
* **Predicted Intent**: `book_hotel`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I can recognize hotel booking requests, but this academic demo does not execute actual room reservations."*
* **Evaluator Talking Point**: Travel service workflow intent recognition.

---

### Scenario 8: Navigation Directions
* **User Spoken Utterance**: *"Give me driving directions to downtown."*
* **Speech Recognition Output**: `give me driving directions to downtown`
* **Predicted Intent**: `directions`
* **Model Confidence**: `89.2%`
* **Chatbot Response**: *"I can recognize requests for navigation and directions, but this academic demo does not connect to live GPS or routing services."*
* **Evaluator Talking Point**: Identifies transit and navigation queries cleanly without inventing fake routes.

---

### Scenario 9: Entertainment / Telling a Joke
* **User Spoken Utterance**: *"Can you tell me a good joke?"*
* **Speech Recognition Output**: `can you tell me a good joke`
* **Predicted Intent**: `tell_joke`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"Why do programmers prefer dark mode? Because light attracts bugs!"*
* **Evaluator Talking Point**: Conversational courtesy and entertainment handling.

---

### Scenario 10: Bot Capability Inquiry
* **User Spoken Utterance**: *"What kind of things can I ask you?"*
* **Speech Recognition Output**: `what kind of things can i ask you`
* **Predicted Intent**: `what_can_i_ask_you`
* **Model Confidence**: `100.0%`
* **Chatbot Response**: *"I am trained to classify 18 intents including weather, alarms, calendar, reminders, financial queries (balance, transfer, bill pay), directions, and dining suggestions."*
* **Evaluator Talking Point**: Explains system features and guides new users through supported operations.

---

### Scenario 11: Low-Confidence / Out-of-Scope Input (Threshold Gating)
* **User Spoken Utterance**: *"Flim flam blorp quux zorp xyz"*
* **Speech Recognition Output**: `flim flam blorp quux zorp xyz`
* **Predicted Intent**: `travel_suggestion (Uncertain)`
* **Model Confidence**: `27.4%` (Below the 50% Threshold)
* **Chatbot Response**: *"I'm not completely sure I understood that. Could you please rephrase or ask in another way?"*
* **Evaluator Talking Point**: **Critical Academic Feature**: Demonstrates that the system does NOT hallucinate or blindly accept low-confidence outputs.

---

### Scenario 12: Silence / Empty Input
* **User Spoken Utterance**: *(Silence / No speech detected)*
* **Speech Recognition Output**: *(Empty string)*
* **Predicted Intent**: `none`
* **Model Confidence**: `0.0%`
* **Chatbot Response**: *"I didn't hear anything. Please speak into the microphone or type your message."*
* **Evaluator Talking Point**: Demonstrates robust defensive error handling against ambient silence or inadvertent clicks.
