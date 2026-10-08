# Introduction to AI & Machine Learning

My personal repo for the **Introduction to Artificial Intelligence and Machine Learning** course
in the Master in Computer Science & Business Technology (MCSBT) at IE University, Term 1, 2026.

I use it to keep my assignments, notes and experiments in one place while I learn.

## Assignments

| # | Topic | Notebook | Status |
| --- | --- | --- | --- |
| I | Working with CSV data in plain Python | [Assignment1.ipynb](AssignmentI/Assignment1.ipynb) | ✅ Done |
| II | Word2vec and semantic similarity of news titles | [Assignment2.ipynb](AssignmentII/Assignment2.ipynb) | ✅ Done |

`AssignmentII/MCSBT_Assignment_2.ipynb` is the word2vec notebook we went through in class. I kept
it as a reference for Assignment II.

### Assignment I: CSV data in plain Python

I loaded a sample of Crunchbase companies (`cb_sample.csv`) with `csv.DictReader`, sorted the rows
by `rank` and counted the unique rows with a `set`. I ran it in Google Colab
([screenshot](AssignmentI/Assignment1_Screenshot.png)).

What I learned:

- `csv.DictReader` gives every row as a dict, but all values are strings, so `rank` has to be
  converted with `int()` to sort numbers instead of text.
- Dicts can't go into a `set`, but `tuple(row.items())` can, which makes de-duplication easy.
- The first column came out as `'\ufeffname'` because the file has a BOM. Opening it with
  `encoding='utf-8-sig'` fixes that.

### Assignment II: Word2vec and semantic similarity

I used Google's pre-trained word2vec model (Google News, 300 dimensions) with `gensim` to find the
news titles most similar to a given title in two webz.io datasets (Amazon and Tesla).

1. Read the titles from JSON Lines files and removed duplicates.
2. Cleaned each title: stripped punctuation and kept only words the model knows.
3. Compared the similarity score with plain `str.split()` against the cleaned version.
4. Scored every title against the chosen one with `n_similarity` and printed the top 50.
5. Turned the steps into functions (`similar_title`, `top_50_similar_news`,
   `display_top_50_news`) that work for any title and dataset.

What I learned:

- A word embedding turns a word into a vector, and words with similar meanings end up close
  together, so similarity becomes cosine similarity between vectors.
- `n_similarity` compares the average vectors of two word lists and quietly skips words the model
  doesn't know. Tokens with punctuation attached (like `"Tesla,"` or `"Q3:"`) count as unknown, so
  they were dropped and the title lost meaning. Stripping punctuation keeps those words and gave
  better scores.
- If none of a title's words are known, the list is empty and `n_similarity` raises
  `ZeroDivisionError`, so empty titles have to be skipped.
- `list(dict.fromkeys(items))` removes duplicates but keeps the original order (a `set` doesn't).
- From the class notebook: vector arithmetic such as `king - man + woman ≈ queen`, and t-SNE to
  plot 300-dimensional vectors in 2D.

## Tools

Python 3.8 · Jupyter · gensim · NumPy · SciPy · pandas · scikit-learn · Matplotlib · Google Colab

## Running the notebooks

I use conda. The versions in [requirements.txt](requirements.txt) are the ones the notebooks were
tested with.

```bash
conda env create -f environment.yml
conda activate ie-ai-ml
jupyter notebook
```

Without conda:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> Keep the pinned versions. `gensim` 4.3 needs `numpy` < 2.0 and `scipy` < 1.14, and the class
> notebook uses `TSNE(n_iter=...)`, which newer scikit-learn versions removed.

### Data

The big files are not in the repo (see [.gitignore](.gitignore)). To run Assignment II again, put
these in `AssignmentII/`:

| File | Size | Source |
| --- | --- | --- |
| `GoogleNews-vectors-negative300.bin` | ~3.6 GB | [Google Drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) (unzip the `.gz`), or `gensim.downloader.load("word2vec-google-news-300")` |
| `webzio_amazon.json`, `webzio_tesla.json` | 61 MB, 22 MB | Course materials (webz.io news) |

Assignment I needs `cb_sample.csv` from the course materials, uploaded to Colab's `/content/` folder.

> [!TIP]
> The full word2vec model needs about 4 GB of RAM. To load only the most common words, use
> `KeyedVectors.load_word2vec_format(path, binary=True, limit=500_000)`.

## Repo structure

```
.
├── AssignmentI/
│   ├── Assignment1.ipynb
│   └── Assignment1_Screenshot.png
├── AssignmentII/
│   ├── Assignment2.ipynb          # my solution
│   └── MCSBT_Assignment_2.ipynb   # class notebook (reference)
├── environment.yml
├── requirements.txt
└── README.md
```

## Course topics

The course goes from classical machine learning to today's AI systems. I'll tick topics off as we
cover them:

- [ ] AI foundations and the technology landscape
- [ ] AI strategy and business transformation
- [ ] Machine learning fundamentals
- [ ] Deep learning
- [ ] NLP and foundation models
- [ ] Large language models
- [ ] Retrieval-augmented generation (RAG)
- [ ] Agentic AI systems
- [ ] Multimodal AI
- [ ] Recommender systems
- [ ] Cloud AI, MLOps and LLMOps
- [ ] Responsible AI and governance
- [ ] Emerging topics
- [ ] Final team project

## Reading list

- Russell & Norvig, *Artificial Intelligence: A Modern Approach*
- Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Foster, *Generative Deep Learning*
- Grainger, *AI-Powered Search*

## AI use

I use AI tools while learning, following the course's AI policy. When AI helped with an assignment,
I say so in that notebook.
