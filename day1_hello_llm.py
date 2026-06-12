from groq import Groq
from dotenv import load_dotenv
import os   

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Configure the Groq client
client = Groq(api_key=api_key)

# Choose the model
# EXPERIMENT 1 — System Prompt
# This controls how the model behaves
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a senior software engineer who explains things concisely in bullet points only."},
        {"role": "user", "content": "Explain what a REST API is."}
    ]
)

print("===Experiment 1: System Prompt===")
print(response.choices[0].message.content)

# EXPERIMENT 2 — TEmperature
# Controls randomness. 0 = deterministic, 1 = creative
response2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Give me a creative name for an AI coding assistant."}
    ],
    temperature=0.5
)

print("\n=== EXPERIMENT 2 - High Temperature ===")
print(response2.choices[0].message.content)

# EXPERIMENT 3 — Token counting
# See how many tokens your request used
response3 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "What is a context window in LLMs?"}
    ]
)
print("\n=== EXPERIMENT 3 - Token Usage ===")
print(f"Input tokens:  {response3.usage.prompt_tokens}")
print(f"Output tokens: {response3.usage.completion_tokens}")
print(f"Total tokens:  {response3.usage.total_tokens}")
print(f"\nResponse: {response3.choices[0].message.content}")