import time, requests

def query_qwen(prompt, batch_size=1, retries=3):
    """
    Query the local Qwen model API with a prompt.

    Args:
        prompt (str): Input text prompt for the model.
        batch_size (int): Number of predictions to generate.
        retries (int): Number of retry attempts if request fails.

    Returns:
        str: Raw response string from the model.
    """
    for _ in range(retries):
        try:
            # Send POST request to local Qwen API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b-instruct",  # Model name
                    "prompt": prompt,                # Input prompt
                    "stream": False,                 # Disable streaming
                    "keep_alive": "30m",             # Keep model alive
                    "options": {                     # Generation options
                        "temperature": 0,            # Deterministic output
                        "num_predict": 20,           # Max tokens to predict
                        "top_p": 0.9,                # Sampling parameter
                        "stop": ["\n"]               # Stop at newline
                    }
                },
                timeout=120,  # Timeout in seconds
            )

            # Raise error if response status is not 200
            response.raise_for_status()

            # Extract "response" field from JSON
            return response.json().get("response", "")

        except:
            # Wait before retrying if request fails
            time.sleep(2)

    # Fallback: return neutral predictions if all retries fail
    return "\n".join(["[0,0,0,0]"] * batch_size)

