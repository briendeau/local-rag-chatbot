import json
from openai import OpenAI
from pathlib import Path

def note_count():
    return len(list(Path("notes").glob("*.txt")))

def multiply(a, b):
    return a * b

def as_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None

def wants_product(text):
    lowered = text.lower()
    has_word = "times" in lowered or "multiply" in lowered or "product" in lowered
    has_digit = any(ch.isdigit() for ch in text)
    return has_word and has_digit

def run_tool(name, args):
    if name == "multiply":
        a = as_number(args.get("a"))
        b = as_number(args.get("b"))
        if a is None or b is None:
            return None
        return str(multiply(a, b))
    if name == "note_count":
        return str(note_count())
    return None

client = OpenAI(
    api_key = "ollama",
    base_url="http://localhost:11434/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers. Use this for any arithmetic product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_count",
            "description": "Return how many .txt files are in the notes folder.",
            "parameters": {"type": "object", "properties": {}},
        }
    }
]

question = input("Ask: ")
messages = [
    {"role": "system", "content": "call multiply for a product of two numbers, call note_count for how many note files, otherwise answer and do not call a tool."},

    {"role": "user", "content": question},
]

for _ in range(3):
    response = client.chat.completions.create(
        model="llama3.2",
        temperature=0,
        messages=messages,
        tools=tools,
    )
    msg = response.choices[0].message 
    if not msg.tool_calls:
        print(msg.content)
        break


    call = msg.tool_calls[0]
    name = call.function.name
    if name == "multiply" and not wants_product(question):
        result = None
    else:
        args = json.loads(call.function.arguments or "{}")
        result = run_tool(call.function.name, args)
        print("tool: ", call.function.name, result)

    if result is None:
        plain = client.chat.completions.create(
            model="llama3.2",
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer in two sentences."},
                {"role": "user", "content": question},
            ],
        )
        print(plain.choices[0].message.content)
        break

    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments or "{}",
            },
        }
    ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": result,
    })
else:
    print("stopped after 3 tool calls.")


