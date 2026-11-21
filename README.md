# Introduction to AI & Machine Learning

Coursework, projects, and experiments from **Introduction to Artificial Intelligence and Machine Learning** in the Master in Computer Science & Business Technology (MCSBT) at IE University (Term 1, 2026).

> [!NOTE]
> This repository serves as the complete course archive (assignments I–IV, midterm, and final group project).

---

## Featured Projects

### 1. Complaint Intelligence for Retail Banks *(Final Term Project)*
> End-to-end NLP & RAG pipeline analyzing **60,000** 2025 CFPB consumer banking complaints — built with Vidak Vukadinovic and Jorge Plaza de los Reyes Álvarez.

- **Pipeline**:
  1. **Campaign Detection**: Sentence embeddings + FAISS cosine similarity + shared-wording confirmation to separate copy-paste complaint campaigns (`16,392` copies identified; `19/20` human-verified pairs).
  2. **Team Routing**: Word2Vec + logistic regression routing complaints to 5 product teams (`82%` test accuracy vs. `72%` keyword rules and `73%` GPT-4o-mini).
  3. **Payout Prioritization**: Word2Vec text + product logistic regression flagging high-risk complaints (`57%` of monetary payouts caught in the top `20%` riskiest queue; `0.0` fairness gap).
  4. **Surge Monitoring**: Tuned LDA topic modeling (`10` topics, coherence `c_v = 0.51`) tracking weekly complaint mixes and catching early surge alarms around real January 2025 banking events.
  5. **Agentic RAG Explainer**: Query-to-filter agent (`Qwen/Qwen3-4B`) over FAISS + routed-team filtering (`Precision@5 = 0.40` vs. `0.36` BM25), evaluated with RAGAs (`faithfulness = 0.98`), LLM-as-judge, and human spot-checks.
- **Files**: [`FinalTermAssignment/Final_Project_Complaint_Intelligence.ipynb`](FinalTermAssignment/Final_Project_Complaint_Intelligence.ipynb) · [`FinalTermAssignment/README.md`](FinalTermAssignment/README.md) · [`FinalTermAssignment/prepare_dataset.py`](FinalTermAssignment/prepare_dataset.py)

---

### 2. RFP Semantic Search with PySpark Word2Vec *(Midterm Project)*
> Scalable Spark MLlib Word2Vec retrieval engine matching sales RFP requirements against **15,952** English EU TED procurement notices (`sinceaihq/eurotender-cpv`).

- **Pipeline**: PySpark DataFrame ingestion & deduplication $\rightarrow$ `RegexTokenizer` & `StopWordsRemover` $\rightarrow$ Spark MLlib `Word2Vec` training (comparing 1 vs. 10 passes) $\rightarrow$ cosine-similarity top-50 notice ranking.
- **Evaluation**: Evaluated Precision@50 and lift across three domain queries (cybersecurity, cloud/database infrastructure, data protection), documenting the jump from 4 to 31 relevant cybersecurity matches after tuning training passes, alongside contextual-averaging limitations.
- **Files**: [`MidTermAssignment/MCSBT_MIDTERM_ASSIGNMENT.ipynb`](MidTermAssignment/MCSBT_MIDTERM_ASSIGNMENT.ipynb) · [`MidTermAssignment/README.md`](MidTermAssignment/README.md)

---

## Assignments

| # | Topic | Key Techniques | Notebook |
| --- | --- | --- | --- |
| I | Working with CSV data in plain Python | `csv.DictReader`, UTF-8 BOM handling, tuple deduplication | [Assignment1.ipynb](AssignmentI/Assignment1.ipynb) |
| II | Word2Vec & semantic similarity | Pre-trained Google News 300d vectors, `gensim`, `n_similarity`, t-SNE | [Assignment2.ipynb](AssignmentII/Assignment2.ipynb) |
| III | Agentic Hybrid RAG & Evaluation | `all-mpnet-base-v2` + FAISS, `BM25Okapi`, Cross-Encoder reranking, `source`/`year` filtering, `Qwen/Qwen3-4B` agent, RAGAs | [Assignment3_RAGAs.ipynb](AssignmentIII/Assignment3_RAGAs.ipynb) ([base](AssignmentIII/Assignment3.ipynb)) |
| IV | Prompt Engineering & Regression Evals | Diagnosing & fixing the NewTelco `gpt-4.1` tool-use prompt, multi-run test suite (`0.89` $\rightarrow$ `0.96` pass rate) | [Assignment4_Woldeyes_Yohannes.ipynb](AssignmentIV/Assignment4_Woldeyes_Yohannes.ipynb) |

### Assignment Highlights

* **Assignment I (`AssignmentI/`)**: Loaded `cb_sample.csv` with `csv.DictReader`, handled UTF-8 BOM (`encoding='utf-8-sig'`), cast numeric ranks, and deduplicated rows via `tuple(row.items())`.
* **Assignment II (`AssignmentII/`)**: Used Google News 300d Word2Vec vectors with `gensim` on webz.io Amazon & Tesla news titles, comparing raw `str.split()` vs. punctuation-cleaned tokens and building reusable top-50 similarity helpers. (`MCSBT_Assignment_2.ipynb` is kept as the in-class reference.)
* **Assignment III (`AssignmentIII/`)**: Built an agentic hybrid RAG pipeline over five landmark LLM papers (Transformer, BERT, GPT-3, LLaMA, DeepSeek-R1) combining `FAISS` (`IndexFlatIP`), `BM25Okapi`, `ms-marco-MiniLM-L-6-v2` cross-encoder reranking, `source`/`year` metadata pre-filtering, `Qwen/Qwen3-4B` retrieval routing, and RAGAs faithfulness/relevancy evaluation.
* **Assignment IV (`AssignmentIV/`)**: Treated a customer-service agent system prompt (`gpt-4.1-2025-04-14`) like production code — auditing baseline failures, refactoring search-keyword and retry rules, and running a 3-attempt regression suite (`0.89` $\rightarrow$ `0.96` mean pass rate) to verify fixes without introducing regressions.

---

## Tech Stack

Python · PySpark (MLlib) · PyTorch · Hugging Face Transformers · Sentence-Transformers · FAISS · BM25 · Gensim · Scikit-Learn · RAGAs · OpenAI API · Pandas · NumPy · Matplotlib · Google Colab

---

## Running the Notebooks

* **Assignments I & II (Local Conda / venv)**:
  ```bash
  conda env create -f environment.yml
  conda activate ie-ai-ml
  jupyter notebook
  ```
  Or with `venv`:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
  pip install -r requirements.txt
  ```
  > [!NOTE]
  > Keep the pinned versions in [`requirements.txt`](requirements.txt) for Assignments I–II (`gensim` 4.3 requires `numpy < 2.0` and `scipy < 1.14`).

* **Assignments III, IV, Midterm & Final Project (Google Colab)**:
  * **Assignment III & Final Project**: Run on a Colab **T4 GPU** runtime (`FinalTermAssignment` uses `data/cfpb_bank_complaints_2025.zip` and `OPENAI_API_KEY` in Colab Secrets for LLM comparisons/RAGAs).
  * **Assignment IV**: Requires `OPENAI_API_KEY` in Colab Secrets (or environment variable).
  * **Midterm**: Upload `MidTermAssignment/rfp_ted_notices.csv` (or unzip `rfp_ted_notices.zip`) to Colab's `sample_data/` folder.

### Large Data & Model Files

Large binaries and raw datasets are excluded via [`.gitignore`](.gitignore):

| File | Size | Where |
| --- | --- | --- |
| `GoogleNews-vectors-negative300.bin` | ~3.6 GB | `AssignmentII/` (`gensim.downloader.load("word2vec-google-news-300")`) |
| `webzio_amazon.json`, `webzio_tesla.json` | 61 MB, 22 MB | `AssignmentII/` (course materials) |
| `cfpb_bank_complaints_2025.csv` & `raw/` | ~77 MB / ~430 MB | `FinalTermAssignment/data/` (built via `prepare_dataset.py`; zipped 20 MB sample included) |

---

## Repo Structure

```text
.
├── AssignmentI/                 # CSV parsing & deduplication in plain Python
├── AssignmentII/                # Word2Vec semantic similarity (gensim)
├── AssignmentIII/               # Agentic Hybrid RAG (FAISS + BM25 + Reranker + RAGAs)
├── AssignmentIV/                # Prompt engineering & regression test suite (GPT-4.1)
├── MidTermAssignment/           # PySpark MLlib Word2Vec RFP semantic search
├── FinalTermAssignment/         # Complaint Intelligence for Retail Banks (End-to-End NLP/RAG)
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Course Topics

- [x] AI foundations and the technology landscape
- [x] AI strategy and business transformation
- [x] Machine learning fundamentals
- [x] Deep learning
- [x] NLP and foundation models
- [x] Large language models
- [x] Retrieval-augmented generation (RAG)
- [x] Agentic AI systems
- [x] Evaluation, prompt engineering, and LLMOps
- [x] Responsible AI and governance
- [x] Final team project

---

## Reading List

- Russell & Norvig, *Artificial Intelligence: A Modern Approach*
- Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Foster, *Generative Deep Learning*
- Grainger, *AI-Powered Search*

---

## AI Use

I use AI tools while learning in line with the course AI policy; usage is documented directly inside the relevant notebooks.
