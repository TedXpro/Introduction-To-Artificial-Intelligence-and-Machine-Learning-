Details


Refer back to the discussion "ROI for AI" discussion where you presented an idea to solve.
Find a dataset similar to the ones attached here from Crunchbase.
You will load your file, similar to the Crunchbase .csv file attached to this assignment.
Similar to the previous assignment, you will use Word2Vec for similarity. However, you will now train a model to predict.
You will write a PySpark program based on the notebook attached which does the following:
Loads your new dataset from your ROI AI idea a Spark dataframe
Cleans up and tokenizes article bodies using the RegexTokenizer and Stopword remover functions provided in the attached notebook.
Trains a Word2Vec model based on the output column produced in step 2.
Implements a sample search query, which can be any combination of words of interest, as shown in attached notebook, and produces the top 50 matching article titles among all article titles


Objectives
After completing this assignment, you will be able to:

Find and load your own dataset for model prediction.
Remove stopwords and tokenize article bodies.
Train a word2vec model that will use a query to find documents and rank them.
Test and Evaluation search queries on your documents, with 50 matching articles.
Assessment
You will be assessed on how well you do the following:



A notebook that Find and load your own dataset for model prediction. (25 points)
A notebook that Remove stopwords and tokenize article bodies. (25 points)
Train a word2vec model that will use a query to find documents and rank them. (25 points)
Test and Evaluation search queries on your documents, with 50 matching articles. (25 points)


Submission
Jupyter Notebook (.ipynb AND .html) with outputs. Feel free to include a comment with your submission(s).
A PDF file of your ROI idea from the discussion board and how you implemented it here.
A link or zip file of the data you used for your program.
