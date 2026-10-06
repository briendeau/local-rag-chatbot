import json
from openai import OpenAI

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
    }
]

question = input("Ask: ")
messages = [
    {"role": "system", "content": "If the user wants a product of two numbers, call multiply. Do not estimate."},
    {"role": "user", "content": question},
]

first = client.chat.completions.create(
    model="llama3.2",
    temperature = 0,
    messages = messages,
    tools=tools,
)

msg = first.choices[0].message

if msg.tool_calls and wants_product(question):
    # parse, call as_number on args, multiply, second model call
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print(args)
    a = as_number(args.get("a"))
    b = as_number(args.get("b"))
    if a is None or b is None:
        print(msg.content or "Not a multiply question.")
    else:
        result = multiply(a, b)
        print("tool ran: ", result)

        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })

        second = client.chat.completions.create(
            model="llama3.2",
            temperature=0,
            messages=messages,
            tools=tools
        )
        print(second.choices[0].message.content)
else:
    plain = client.chat.completions.create(
        model="llama3.2",
        temperature=0,
        messages=[
            {"role": "system", "content": "Answer in two sentences."},
            {"role": "user", "content": question},
        ],
    )
    print(plain.choices[0].message.content)