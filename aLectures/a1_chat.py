# local model, local inference, private data
# a1_chat.py  →  http://localhost:11434/v1  →  llama3.2

from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
    )

messages = [
    {"role": "system", "content": "You answer in 3 short sentences max."}
]

while True:
    question = input("Ask something: (type quit to stop) ")
    if question.strip().lower() == "quit":
        break

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama3.2:latest",
        temperature=0.2,
        messages=messages,
    )

    reply = response.choices[0].message.content
    print(reply)
    messages.append({"role": "assistant", "content": reply})
    print(messages)



