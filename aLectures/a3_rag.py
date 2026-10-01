import math
from openai import OpenAI


def embed(text):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text,
    )
    return response.data[0].embedding

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

notes = [
    "JSON is a data serialization format, a text string of characters representing a data structure in a native programming language.",
    "You turn your object data into json with json.dumps(data)",
    "Ollama server hosts and serves the models at the url http://localhost:11434/v1",
    "different models use different server endpoints like /chat/completions or /embeddings",
    "search requires embeding notes and a question then score the question vs each note using cosine similiarity.",
    "Two pointers means two indexes on a list, not C pointers.",
    "An embedding is a vector that represents meaning.",
    "Player objects are saved as JSON dicts using to_dict and from_dict.",
]

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
    )

notes_vec = []

for note in notes:
    notes_vec.append((note, embed(note)))

query = input("Ask a question: ")
q_vec = embed(query)

# scoring
scored = []
for note, vec in notes_vec:
    score = cosine(q_vec, vec)
    scored.append((score, note))


# sort reverse and print top 3
top_three_notes = []
scored.sort(reverse=True)
for score, note in scored[:3]:
    print(round(score, 3), note)
    top_three_notes.append(note)

# put top 3 strings into llama3.2 prompt now
# finally print prompt answer
combined_string = ("Context:\n" + "\n".join(top_three_notes)
                    + "\nQuestion: " + query)

print(combined_string)

# build message list with system and user roles
messages = [
    {"role": "system", "content": "use the full relevant context sentence, not a three-word summary, if the question can't be answered there, reply with exactly DON'T KNOW and nothing else."},
    {"role": "user", "content": combined_string},
    ]

response = client.chat.completions.create(
    model="llama3.2:latest",
    temperature=0,
    messages=messages,
)

print(response.choices[0].message.content)





    