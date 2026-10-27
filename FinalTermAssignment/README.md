Final Project Presented In Class
Each student group will deliver a live presentation (no more than 12 PowerPoint slides), covering:
Definition of use case and background research
Data source specification and procurement details
Design choices and the rationale for the implemented methodologies
Evaluation results and/or demonstration
Findings and conclusions
Student groups will be assessed on how clearly they present their project, facilitate class discussion and feedback. The instructor's frame of reference is a situation when students are pitching their innovative analytics idea to non-technical decision makers or executives. The presentation must inspire confidence and persuade the audience that the project is worth investing organizational time and resources.
There are THREE Components to this Final Project: A presentation, coding (and dataset), and final peer review.
Groups of 2/3 people max to be added to the discussion board.
You will be given LIVE feedback by the professor immediately after your presentation to incorporate to your final solution that you will delivery here.
Objectives
After completing this assignment, you will be able to:
Design and deliver an impactful presentation that explains AI outputs to a non-technical, business audience
Synthesize and leverage all your class learnings in your Term Project
Implement an analytics project leveraging AI methods
Gain proficiency in developing an end-to-end scalable Python or PySpark application
Evaluate their implementation against the proposed success criteria
Python Implementation and Dataset
For this assignment, a student group must implement an end-to-end text AI application articulated in the Term Project Presentation. The implementation shall be developed in Python in either Jupyter Notebook/Lab or Google Colab. It should leverage a text dataset and at least three AI methods (e.g. LLMs, Topic Modeling, RAG, etc.) covered in the course. Student groups are encouraged to reuse the Python code modules that they have developed for their individual assignments.
Students can use datasets provided in the course assignments or obtain another text dataset from available public or commercial data sources, such as the below:
[Data.govLinks to an external site](https://www.data.gov/)
[Open Data NYLinks to an external site](https://data.ny.gov/)
[LegiscanLinks to an external site](https://legiscan.com/datasets)
*Kaggle datasets need to be discussed with instructor ahead of time to ensure the authenticity of the solution developed for the project. Below is a sample general scope for a term project implementation.
Example Python Implementation Ideas
Import a news dataset of 10,000+ articles (in JSON format) about a chosen entity (organization, person, or country) into Python or Spark Data frame
Deduplicate news articles using tokenization and train a Word2Vec, GloVe, FastText or BERT word embedding model
Train an LDA topic clustering model on texts of the obtained news articles using Gensim, Scikit-learn or Spark MLLib Python packages. Tune the parameters of LDA, such as n-gram values, minimum and maximum document frequency thresholds.
Create a RAG pipeline that will consume the articles using the notebook from class and allow it to query your articles returning back the closest article with FAISS.
Construct a topic keyword taxonomy in JSON format and design a Python function to use the trained word embedding model for classifying input texts against a taxonomy.
Classify article texts and titles using the taxonomy and aggregate results over time to produce highest-ranking (evolving) topics.
There are two previous submissions attached, a low-scoring one and a high-scoring one. Please try to understand why one scored lower than the other -- they are both good business ideas but something is missing in the technical implementation.
Assessment
You will be assessed on how well you present your Term Project and on the quality of your slide deck (i.e. does it reinforce the oral presentation, enhance audience understanding, and provide concrete takeaways?). The instructor's frame of reference will be as if you were presenting this Project as an innovative idea to decision-makers within an organization - would the presentation inspire confidence that the activities in your Project are worth undertaking and investing organizational time and resources in?
Student groups will be assessed on the correctness of the implemented methodologies and on the overall alignment of the implementation with the proposed analytics use case. Specific criteria is outlined in the grading rubric below.
Important! Please make sure that you have an evaluation method for your model. You have to show that the model would work for your business case.
Project Peer Feedback
Project Peer Feedback is a graded individual assignment, part of the Term Project. For this assignment, you must submit a maximum 150-word comment that must detail your and your team members' specific contributions to the project. It is understandable that while every group member should contribute equally to the project, some members may contribute more or less than expected. This survey is to identify the rare cases when students did not meet the usual standard of participation. Please, include your Group # in the submission.
Submission
A powerpoint deck (not a pdf) of the slides that you presented AFTER the feedback given in class.
A Jupyter Notebook (.ipynb and .html) of the code your group used to finalize the project.
A PDF file (not a word document) of your peer feedback described above.
NOTE: While this is a group project. Each person is responsible for uploading ALLL three of the requirements.
NOTE: Students often times wait until the last minute, don't wait. Also, there are several facets that should show up in your presentation like: (1) the business topic you chose (2) the dataset you used (3) the three approaches you used from the class and (4) the evaluation method you used.