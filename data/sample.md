# Retrieval-Augmented Generation (RAG): A Short Primer

Retrieval-augmented generation, or RAG, is a technique that lets a large language model answer questions using documents it was never trained on. Instead of relying only on what the model memorised, the system first looks up relevant passages and gives them to the model as context.

## How a RAG pipeline works

The pipeline has two phases. During ingestion, documents are loaded and split into chunks of a few hundred tokens. Each chunk is converted into an embedding, a vector of numbers that captures its meaning, and stored in a vector database such as Qdrant. During querying, the user's question is embedded with the same model, the database returns the chunks whose vectors are most similar, and those chunks are placed in the prompt so the language model can write a grounded answer.

## Why chunk size matters

Small chunks are precise but can lose surrounding context. Large chunks keep context but dilute the embedding and cost more tokens. A common starting point is 512 tokens with a 50-token overlap, so that a sentence cut at a boundary still appears whole in one of the neighbouring chunks.

## Benefits and limits

RAG reduces hallucination, lets answers cite their sources, and allows knowledge to be updated by simply re-ingesting documents, with no model retraining. Its quality depends on retrieval: if the right chunk is not retrieved, the model cannot use it.
