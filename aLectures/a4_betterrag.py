from pathlib import Path
import math
from openai import OpenAI

def load_notes(folder):
    texts = []
    for path in Path(folder).glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return texts

def make_chunks(note_strings, size=300, overlap=50):
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    chunks = []
    for note in note_strings:
        start = 0
        while start < len(note):
            chunk = note[start:start+size]
            if (chunk):
                chunks.append(chunk)
            start = start + (size - overlap)
    return chunks

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

notes_dir = Path("notes")

if not notes_dir.exists():
    print("notes folder missing")
    raise SystemExit
elif len(load_notes(notes_dir)) == 0:
    print("folder exists but no .txt files")
    raise SystemExit
else:
    note_strings = load_notes(notes_dir)
    chunks = make_chunks(note_strings)
    print(len(note_strings), len(chunks))

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
    )

notes_vec = []
for chunk in chunks:
    notes_vec.append((chunk, embed(chunk)))

query = input("Ask a question: ")
q_vec = embed(query)

# scoring
scored = []
for chunk, vec in notes_vec:
    score = cosine(q_vec, vec)
    scored.append((score, chunk))


# sort reverse and print top 3
top_three_notes = []
scored.sort(reverse=True)
for score, chunk in scored[:3]:
    print(round(score, 3), chunk)
    top_three_notes.append(chunk)

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

    