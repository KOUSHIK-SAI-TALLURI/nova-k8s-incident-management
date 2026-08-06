import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are NOVA, an intelligent DevOps incident management assistant."
        },
        {
            "role": "user",
            "content": "A pod is crashing with OOMKilled error. What could be the root cause?"
        }
    ],
    temperature=0.1,
    max_tokens=500
)

print("✅ Groq connected successfully")
print("\nNOVA Response:")
print(response.choices[0].message.content)