from fastapi import FastAPI
from pydantic import BaseModel
from a4_betterrag import load_notes, make_chunks, embed, cosine
from openai import OpenAI

client = OpenAI(
    api_key = "ollama",
    base_url="http://localhost:11434/v1",
)

notes = load_notes("notes")
chunks = make_chunks(notes)

# loop through chunks and embed each one and put it next to the chunk.
notes_vec = []
for chunk in chunks:
    notes_vec.append((chunk, embed(chunk)))



app = FastAPI()

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Ask):
    # what is true for the whole server and what is true for this question ask. follow the data movement path.
    # load and embed the notes once
    # embed body.question, score chunks, take top 3, one create,
    # no loop, system lime answer from context only, or "DON'T KNOW", no input() the question is in the body. no tools.
    # question needs embeded each time ask is called.
    q_vec = embed(body.question)

    # scoring happens inside ask because we compare q_vec score against the chunk score
    scores = []
    for chunk, vec in notes_vec:
        score = cosine(q_vec, vec)
        scores.append((score, chunk))

    # sort reverse return top 3
    top_three = []
    scores.sort(reverse=True)
    print(scores[:3])
    for score, chunk in scores[:3]:
        print(round(score, 3), chunk)
        top_three.append(chunk)

    context_string = "\n".join(top_three) + "\nQuestion: " + body.question
    print(context_string)

    messages = [
        {"role": "system", "content": "Answer only from the context, otherwise reply with exactly DON'T KNOW and nothing else."},
        {"role": "user", "content": context_string}
        ]

    response = client.chat.completions.create(
        model="llama3.2",
        temperature=0,
        messages=messages,
    )

    
    return {"answer": response.choices[0].message.content,
            "chunks": top_three,
            }




