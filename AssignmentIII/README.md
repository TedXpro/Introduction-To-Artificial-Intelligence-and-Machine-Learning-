# Assignment III — Retrieval Augmented Generation (RAG) Pipeline

## Details

1. Refer back to the slides on Retrieval Augmented Generation (RAG) and pipelines.
2. For this assignment, you will use Colab to create your notebook and you will need an OpenAI Key if you don't have one already (https://platform.openai.com/api-keys). This tutorial may help: https://pickaxe.co/post/how-to-get-your-openai-api-key-a-step-by-step-guide
3. You will create a RAG pipeline using your own documents. Feel free to use the previous assignment files or any files of your choice (note: you will have to extract the text from the file).
4. Switch out the default model for different models from Hugging Face. `all-mpnet-base-v2` is slower but more accurate than `all-MiniLM-L6-v2`. The original model is found here: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
5. **Add Metadata Filtering:** Store a `source`/`date` field per chunk and filter the FAISS search by it before ranking.
6. **Add an agent:** Let the model decide whether to retrieve, and with what query, instead of always retrieving — this is the first step toward an agentic RAG system.
7. You are provided a notebook that builds a complete RAG pipeline with a:
   - **Knowledge base** — 6 short passages, deliberately built from this course's own material (Transformers, pretraining, fine-tuning, RLHF, reasoning models, evaluation) — so students can query the session itself
   - **Chunking** — overlapping word-bounded chunks
   - **Embeddings & index** — sentence-transformers (`all-MiniLM-L6-v2`) + FAISS (`IndexFlatIP` for cosine similarity)
   - **Retrieval** — top-k nearest chunks by cosine similarity
   - **Generation** — a prompt that restricts the model to the retrieved context only; defaults to a free local model (`flan-t5-base`) with a one-line switch to the OpenAI API for better answers
   - **Full pipeline** — retrieve + generate combined, demoed on 3 questions (including one the knowledge base can't answer, to confirm the model says so instead of hallucinating), plus an optional RAGAs faithfulness-check cell tying back to the evaluation slides

## Objectives

After completing this assignment, you will be able to:

- Create a RAG pipeline using your own documents
- Use your choice of Hugging Face model for the brain of your pipeline
- Rank documents using the FAISS method
- Add an agent to a RAG pipeline

## Assessment

You will be assessed on how well you do the following:

| # | Criterion | Points |
|---|-----------|--------|
| 1 | A notebook that has a runnable RAG pipeline with your own documents | 25 |
| 2 | A notebook that uses a Hugging Face model for its LLM (not the default one) | 25 |
| 3 | A notebook that uses a source/date field for ranking documents via FAISS | 25 |
| 4 | A notebook that uses an agent for the RAG pipeline | 25 |

## Submission

- Jupyter Notebook (`.ipynb` **and** `.html`) with outputs. Feel free to include a comment with your submission(s).
