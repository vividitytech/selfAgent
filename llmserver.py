      
      
from flask import Flask, request, jsonify
import os
import torch
import logging
import openai  # OpenAI API
import numpy as np
from transformers import AutoTokenizer, AutoModel
from behaviorclassifier import BehaviorClassifier
from utils import load_dict_from_txt
from configs import CONFIGS as config

class LLMApiService:
    """
    A class to encapsulate the LLM API service with conversation history.
    """

    def __init__(self, openai_api_key, openai_model_name, device, max_history_length=10, max_conversation_length=3000, model_path = "./trained_model", num_personality=13, learning_rate=0.01):
        """
        Initializes the LLMApiService.

        Args:
            openai_api_key (str): The OpenAI API key.
            openai_model_name (str): The OpenAI model name.
            max_history_length (int): The maximum length of the conversation history (number of turns).
            max_conversation_length (int): Maximum length of conversation in terms of characters.
        """
        self.openai_api_key = openai_api_key
        self.openai_model_name = openai_model_name
        self.max_history_length = max_history_length
        self.max_conversation_length = max_conversation_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classifier = BehaviorClassifier.from_pretrained(model_path,num_classes = num_personality) #load model for prediction
        self.classifier.to(device)
        labelmaps = load_dict_from_txt(model_path+"/labels.txt")
        self.labelmaps = {value: key for key, value in labelmaps.items()}
        self.num_personality = num_personality
        self.learning_rate = learning_rate
        self.global_characteristics = np.zeros(self.num_personality)  # vector with num_personality to describle global characteristics
        self.personalized_characteristics = {}
        self.conversation_history = {}  # Conversation History (Dictionary keyed by IP address)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime%)s - %(levelname)s - %(message)s')



    def get_client_ip(self):
        """Gets the client IP address from the request."""
        return request.remote_addr

    def get_conversation_history_for_ip(self, ip_address):
        """Retrieves the conversation history for a given IP address."""
        if ip_address not in self.conversation_history:
            self.conversation_history[ip_address] = []
        return self.conversation_history[ip_address]

    def get_personalized_characteristics_for_ip(self, ip_address):
        """Retrieves the personalized characteristics for a given IP address/user."""
        if ip_address not in self.personalized_characteristics:
            self.personalized_characteristics[ip_address] = np.zeros(self.num_personality)
        return self.personalized_characteristics[ip_address]
    
    def update_conversation_history(self, ip_address, role, content):
        """Updates the conversation history for a given IP address."""
        history = self.get_conversation_history_for_ip(ip_address)
        history.append({"role": role, "content": content})

        # Keep history within the maximum length (number of turns)
        if len(history) > self.max_history_length:
            history.pop(0)  # Remove the oldest message

        # Check conversation length and remove oldest messages if it exceeds the threshold
        while self.estimate_conversation_length(history) > self.max_conversation_length and len(history) > 0:
          history.pop(0)

        self.conversation_history[ip_address] = history

    def update_characteristics(self, ip_address, role, content):
        if role=="user":
            personalized_characteristics = self.get_personalized_characteristics_for_ip(ip_address)
            prob, label =  self.classifier.predict(content, self.tokenizer, device=device)
            # update characteristics for each user
            personalized_characteristics = personalized_characteristics + self.learning_rate * prob[0]
            arr_sum = np.sum(personalized_characteristics)
            self.personalized_characteristics[ip_address] = personalized_characteristics/arr_sum
            # update the global characteristics
            self.global_characteristics = self.global_characteristics  + self.learning_rate * prob[0]
            # normalize 
            arr_sum = np.sum(self.global_characteristics)
            self.global_characteristics = self.global_characteristics/arr_sum
            return self.global_characteristics

    def generate_system_message(self, ip_address, type="global"):
        if type =="global":
            idx_sorted_desc = np.argsort(self.global_characteristics)[::-1]
        else:
            personalized_characteristics = self.get_personalized_characteristics_for_ip(ip_address)
            idx_sorted_desc = np.argsort(personalized_characteristics)[::-1]
        system_message = """you have the following characteristics which scales from 1 to 10, and when you answer user questions, 
        you should respond with corresponding scales. For example, if you have patientness scale with 10, then you should answer user'question very patiently. 
        If you have patientness scale of 1, then you would not like to answer user'question, or even no response.\n"""
        
        
        for idx in idx_sorted_desc:
            if self.global_characteristics[idx]>=0.06:
                scale = np.round(10*self.global_characteristics[idx])
                characteristic = self.labelmaps[idx]
                system_message = system_message  +  " You have characteristics " + characteristic + " with "+ str(scale) +"\n"#f"you have {characteristic} with scale {scale} \m"
            else:
                break
        
        return system_message

    def estimate_conversation_length(self, history):
        """Estimates the length of the entire conversation (in characters)."""
        total_length = 0
        for message in history:
            total_length += len(message["content"]) #Simply use the length of string
        return total_length

    def generate_text(self, prompt):
        """
        Generates text using the OpenAI API, considering conversation history.

        Args:
            prompt (str): The user's prompt.

        Returns:
            tuple: A tuple containing a boolean indicating success, the generated text (if successful), and an error message (if unsuccessful).
        """
        ip_address = self.get_client_ip()
        history = self.get_conversation_history_for_ip(ip_address)

        # update characteristics
        self.update_characteristics(ip_address, "user", prompt)
        # generate system message
        system_message = self.generate_system_message(ip_address)
        # Construct the messages for the OpenAI API, including history and new prompt
        messages = [{"role": "system", "content": f"You are a helpful assistant with {system_message}. Remember the previous conversation."}] + history + [{"role": "user", "content": prompt}]

        try:
            openai.api_key = self.openai_api_key
            if not openai.api_key:
                return False, None, "OPENAI_API_KEY not set in environment variables."

            response = openai.chat.completions.create(
                model=self.openai_model_name,
                messages=messages,
                max_tokens=200,  # Adjust as needed
                n=1  # Adjust as needed
            )

            generated_text = response.choices[0].message.content
            logging.info(f"Generated text from OpenAI for IP {ip_address}: {generated_text[:50]}...")

            # Update conversation history with user prompt and assistant response
            self.update_conversation_history(ip_address, "user", prompt)
            self.update_conversation_history(ip_address, "assistant", generated_text)

            return True, generated_text, None

        except Exception as e:
            logging.error(f"Error during text generation with OpenAI: {e}")
            return False, None, str(e)


    def get_history(self):
      """
      Retrieves the conversation history for the requesting IP address.
      """
      ip_address = self.get_client_ip()
      history = self.get_conversation_history_for_ip(ip_address)
      return history



# -------------------- Flask Application --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use CUDA if available
app = Flask(__name__)

# Initialize the LLM API service
try:
    openai_api_key = config.get("chatconfig", None)["api_key"] #os.environ.get("OPENAI_API_KEY")
    openai_model_name = config.get("chatconfig", None)["model"]  # Or your model choice
    max_history_length = 10
    max_conversation_length = 3000
    llm_service = LLMApiService(openai_api_key, openai_model_name, device, max_history_length, max_conversation_length)
except Exception as e:
    print(f"Error initializing LLM service: {e}") #Using print because logger might not be ready in this time.
    llm_service = None #Important: set to None

@app.route('/generate', methods=['POST'])
def generate_text_route():
    """
    Flask route for generating text.
    """
    if llm_service is None:
        return jsonify({"error": "LLM service failed to initialize.  Check server logs."}), 500

    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data["prompt"]

        success, generated_text, error_message = llm_service.generate_text(prompt)

        if success:
            return jsonify({"generated_text": generated_text})
        else:
            return jsonify({"error": f"Error generating text: {error_message}"}), 500

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": f"Error processing request: {e}"}), 500


@app.route('/history', methods=['GET'])
def get_history_route():
    """
    Flask route for getting the conversation history.
    """
    if llm_service is None:
        return jsonify({"error": "LLM service failed to initialize.  Check server logs."}), 500

    try:
        history = llm_service.get_history()
        return jsonify({"history": history})

    except Exception as e:
        logging.error(f"Error getting history: {e}")
        return jsonify({"error": f"Error getting history: {e}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "ok"}), 200

# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "hello"}' http://localhost:5000/generate
# -------------------- Main --------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get port from environment or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)  # Listen on all interfaces for Docker

    
