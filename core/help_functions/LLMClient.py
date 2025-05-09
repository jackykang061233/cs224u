import openai
import os
import json

class LLMClient:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Initializes the LLMClient.

        Args:
            api_key (str): Your OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model (str): The OpenAI model to use (default is 'gpt-4o-mini').
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY env var or pass api_key argument.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def call_llm(self, system_prompt, user_prompt):
        """
        Sends a chat completion request to the API.

        Args:
            system_prompt (str): The system-level prompt (defines assistant behavior).
            user_prompt (str): The user message.

        Returns:
            str: The assistant's reply.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error communicating with OpenAI API: {e}")
            return None

