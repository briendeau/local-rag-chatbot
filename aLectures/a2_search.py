# local model, local inference, private data
# a1_chat.py  →  http://localhost:11434/v1  →  llama3.2
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


note_vecs = []
for note in notes:
    # each element is a tuple (text, vector)
    note_vecs.append((note, embed(note)))

query = input("ask a question (search): ")
qvec = embed(query)



# score and print top 3 scores

scored = []
for note, vec in note_vecs:
    score = cosine(qvec, vec)
    scored.append((score, note))

scored.sort(reverse=True)
for score, note in scored[:3]:
    print(round(score, 3), note)




