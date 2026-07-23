# Pramaan

A RAG system for Indian law, built to actually understand retrieval-augmented generation properly instead of doing the "embed a PDF, ask questions" tutorial version.

Pramaan means "proof" in Hindi. Every answer cites the exact document and page it came from, and if the corpus doesn't actually cover something, it says so instead of making it up. I picked legal text on purpose because citations actually matter here, nobody cares if an AI hallucinates about movie trivia, but a made-up legal answer is a real problem.

## What it does

```
You: What is the punishment for murder under the Bharatiya Nyaya Sanhita?

Pramaan: Under Section 103 of the BNS, whoever commits murder shall be
punished with death or imprisonment for life [1]. Murder is defined in
Section 101 as culpable homicide committed with intent to cause death [2].

Sources: [1] bns.pdf, p.33  [2] bns.pdf, p.32
```

*(screenshot/gif of the chat UI goes here)*

## How it works

- Query comes in -> if it clearly names a specific Act, restrict search to just that document (learned this the hard way, see below)
- Run BM25 (keyword) and vector search (BAAI/bge-large-en-v1.5) in parallel, merge results with Reciprocal Rank Fusion
- Rerank the merged candidates with a cross-encoder for precision
- If the best score is too low, refuse - don't even call the LLM
- Otherwise, build a prompt from the retrieved chunks and generate a cited answer

BM25 + vector search together because vector search misses exact terms (like a specific article number) that keyword search nails instantly, and vice versa for anything meaning-based.

## The docs

20 documents, all pulled directly from official government sites and trusted sources on Indian Law (`legislative.gov.in`, `mha.gov.in`, `indiacode.nic.in`, `meity.gov.in`) Constitution of India, the 2023 criminal codes (BNS/BNSS/BSA), the 4 Labour Codes, core civil law (Contract Act, CPC, Transfer of Property, etc.), and practical stuff like the Motor Vehicles Act and RTI Act.

## Does it actually work?

Wrote 122 test questions covering every document, verified every expected answer against the actual government PDFs myself. Tested answer rate, refusal precision (does it correctly say "I don't know"), and citation rate.

Also ran the same questions against 4 different LLMs instead of just picking one:

| Model | Answer Rate | Refusal Precision | Citation Rate | Latency |
|---|---|---|---|---|
| **Nemotron 3 Ultra** (chosen) | 76-81% | **100%** | **100%** | ~8-10s |
| Nemotron 3 Super | 96.7% | 90% | 94.8% | ~4.2s |
| Nemotron 3 Nano | 75% | **100%** | **100%** | ~1.7s |
| Llama 3.3 / Qwen3-Next | - | - | - | kept timing out |

Went with Ultra over Super because Super answering more questions came at the cost of confidently answering 1 in 10 things it shouldn't have. Given citations are the whole point, that felt like the wrong tradeoff.

## Bugs that taught me the most

- **Article 14 was unretrievable.** Turned out my chunker was cramming 4 short constitutional articles into one chunk since they all fit under my token limit together. Article 14 is one sentence, buried among 3 other articles, its embedding got diluted and stopped "pointing at" equality-before-law at all. Fixed by hard-splitting on section boundaries instead of just token count.
- **That fix broke other things.** Some sections used a slightly different numbering format (`"11 . Res judicata"` with a space) that my new regex missed, silently merging two sections back together. Footnote markers like `"2. Ins. by s. 94, ibid."` were also getting misread as real sections.
- **Silent embedding truncation.** I'd measured chunk sizes with `tiktoken` this whole project, but my embedding model has its own tokenizer with a hard 512-token limit and counts differently. Chunks I thought were safely sized were getting silently truncated on embedding, no error, nothing.
- **A reasoning model leaking its own thoughts into answers.** Nemotron 3 Ultra reasons by default, and I didn't know the flag to disable it was different from an older model generation's convention. Answers were literally printing "the user is asking about X, let me check..." until I found the right parameter.

## Stack

LangChain · ChromaDB · BAAI/bge-large-en-v1.5 · cross-encoder/ms-marco-MiniLM-L-6-v2 · rank-bm25 · Nemotron 3 Ultra (NVIDIA NIM) · FastAPI · Next.js · GitHub Actions for CI

## Running it

```bash
git clone https://github.com/MeloveGupta/legal-rag.git
cd legal-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add NVIDIA_API_KEY, free at build.nvidia.com

python -m src.legal_rag.ingestion.ingest --all --reset
uvicorn main:app --port 8000

cd frontend && npm install && npm run dev
```

Eval: `python evals/scripts/evaluate.py`

## Why no live link

This needs an embedding model + reranker + BM25 index all loaded in memory at once, realistically 2GB+ RAM, running constantly. Every genuinely free host I found (Render, Back4app, Koyeb) caps out around 256-512MB. Everything with enough RAM (Hugging Face Spaces, Cloud Run, Oracle) wants payment.

## If I kept going

- Real case law via the Indian Kanoon API (has a workable free tier, just didn't get to it)
- Family law statutes - held off since India has multiple personal law systems depending on religion, didn't want to bolt that on carelessly
- Company law (Companies Act, Arbitration Act)

## License

MIT