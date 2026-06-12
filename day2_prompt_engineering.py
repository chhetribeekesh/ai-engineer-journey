from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "review this code: def add(a,b): return a+b"}
    ]
)
print("=== BEGINNER PROMPT ===")
print(response.choices[0].message.content)

# SENIOR PROMPT — specific, consistent, controlled
senior_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",    
    messages=[
        {"role": "system", 
         "content": """You are an expert pythopn code reviewer with experience more than 10 years experience.
                When give a code you always:
                1. Identify specific issues with line references
                2. Rate severity as: low, medium, or high
                3. Suggest a concrete fix
                4. Keep feedback under 5 bullet points
                5. Never give generic advice
            """
         },
         {
             "role": "user",
             "content": "review this code: def add(a,b): return a+b"
         }

    ]
)
print("\n=== SENIOR PROMPT ===")
print(senior_response.choices[0].message.content)


# FEW-SHOT PROMPTING
# We teach the model the exact output pattern we want

few_shot_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system",
          "content":"You classify code review feedback into categories."
        },
        # Example 1 — you show input
        {"role": "user", "content": "No type hints on function parameters"},
        # Example 1 — you show expected output
        {"role": "assistant", "content": "Category: STYLE | Severity: LOW"},

         # Example 2
        {"role": "user", "content": "SQL query is vulnerable to injection"},
        {"role": "assistant", "content": "Category: SECURITY | Severity: HIGH"},

        # Example 3
        {"role": "user", "content": "Function has no error handling"},
        {"role": "assistant", "content": "Category: RELIABILITY | Severity: MEDIUM"},

         # Now the real question — model follows the pattern
        {"role": "user", "content": "Passwords are stored in plain text"}
    ]
)
print("\n=== FEW-SHOT PROMPT ===")
print(few_shot_response.choices[0].message.content)

# STRUCTURED JSON OUTPUT
json_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", 
         "content": """ 
            You are a Python code reviewer.
            You MUST respond with ONLY a JSON object, no explanation, no markdown, no backticks.
            Exact format:
            {
                "issues": ["issue1", "issue2"],
                "severity": "low | medium | high",
                "suggested_fix": "your fix here",
                "score": <number 1-10>,
                "summary": "one line summary"
            }"""
        },
        {
            "role": "user",
            "content": """
            Review this code:
            def get_user(id):
                query = "SELECT * FROM users WHERE id = " + id
                return database.execute(query)
            """
        }
    ],
    temperature=0.1 # low temperature = consistent structured output
)
print("\n=== STRUCTURED JSON OUTPUT ===")
raw = json_response.choices[0].message.content
print(raw)

# Now parse it into a Python dictionary
parsed = json.loads(raw)
print("\nParsed and accessible:")
print(f"Score:     {parsed['score']}/10")
print(f"Severity:  {parsed['severity']}")
print(f"Summary:   {parsed['summary']}")
print(f"Issues:    {parsed['issues']}")
print(f"Fix:       {parsed['suggested_fix']}")