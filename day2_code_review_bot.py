from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def review_code(code: str) -> dict:
    """
    Takes a code snippet and returns a structured review.
    This is the kind of function you'd expose as an API endpoint.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": """You are an expert code reviewer.
Respond ONLY with a JSON object, no markdown, no backticks, no explanation.
Exact format:
{
    "issues": ["issue1", "issue2"],
    "severity": "low | medium | high",
    "suggested_fix": "your fix here",
    "score": <number 1-10>,
    "summary": "one line summary"
}"""
            },
            # Few-shot examples so model knows exactly what we expect
            {"role": "user", "content": "def add(a, b): return a + b"},
            {"role": "assistant", "content": '{"issues": ["no type hints", "no docstring"], "severity": "low", "suggested_fix": "def add(a: int, b: int) -> int:", "score": 7, "summary": "Simple function, minor style issues"}'},

            # Real input
            {"role": "user", "content": f"Review this code:\n{code}"}
        ]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # This is important — always handle LLM output failures gracefully
        return {"error": "Model returned invalid JSON", "raw": raw}


def print_review(code: str, review: dict):
    print("\n" + "="*50)
    print("CODE REVIEWED:")
    print(code)
    print("\nREVIEW RESULT:")
    if "error" in review:
        print(f"Error: {review['error']}")
        return
    print(f"  Score:    {review['score']}/10")
    print(f"  Severity: {review['severity'].upper()}")
    print(f"  Summary:  {review['summary']}")
    print(f"  Issues:")
    for issue in review['issues']:
        print(f"    - {issue}")
    print(f"  Fix: {review['suggested_fix']}")
    print("="*50)


# Test with 3 different code snippets
code1 = """
def get_user(id):
    query = "SELECT * FROM users WHERE id = " + id
    return db.execute(query)
"""

code2 = """
def calculate_discount(price, discount):
    return price - (price * discount / 100)
"""

code3 = """
import pickle
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
"""

for code in [code1, code2, code3]:
    review = review_code(code)
    print_review(code, review)