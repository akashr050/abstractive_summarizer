# Abstractive Summarization and Extrinsic Evaluation via Q/A
Text summarization is a process to create a representative summary or abstract of the entire document, by finding the most informative portions of the article. There are two approaches for automatic summarization: extractive summarization and abstractive summarization. The current techniques to evaluate an summarizer are BLEU and Rogue-n scores. These metrics are based on the overlap between the predicted summaries and the summaries provided by human (generally mechanical turks or news headlines). These metrics can be good system to evaluate the extractive summaries because they extract word features from the input text. Hence, we expect there to be a huge overlap between the predicted and human-provided summaries. For abstractive summarizer which aims to understand the text and provide a summary, it is not necessary for them to have the same words as there are in the human-provided summaries. But due to non-availability of a better metric system, we are still using BLEU and Rogue-n scores to evaluate abstractive summaries. Our understanding, is that if a summary can answer the questions based on the text then it is a good summary. Hence, we propose to use the Question/Answering system as an evaluation metric to evaluate the summaries. 

## Getting Started

In order to run the code for training and evaluation following are the required steps:
1) Clone this github repository on your local machine
2) Create a directory named 'workspace' in the cloned repository
3) Download the pickle files for training from https://umich.box.com/shared/static/w96f785uzl5x3vrut8nyyptcupjvuiou.pkl , https://umich.box.com/shared/static/32khbsbdhnib2xmuupty85yllsgcsdxv.pkl , https://umich.box.com/shared/static/qxzq3ozlzo0gr9ywvjip8yg87xopbizj.pkl and save in the workspace directory
4) Create a virtual python environment and install the following packages: scikit-learn, tensorflow-1.4, scipy
5) Activate the virtual environment and run python3 basic_rnn_summarizer.py to train the model.
```
python3 basic_rnn_summarizer.py
```
6) In order to evaluate the model, run python3 basic_rnn_evaluation.py
```
python3 basic_rnn_evaluation.py
```
7) This evaluation script will save the model outputs as title_out.txt file in your workspace directory. 
8) 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Authors
Akash Rastogi, Bhavika Jalli Reddy, Vidya Mansur, Daniel D'souza

## Acknowledgments
The team members would like to thank Prof. Rada Mihalcea for her insights about the approach of the project and the GSI's Steven Wilson, Laura Wendlandt for their constant assistance and timely reviews which played a vital role in the completion of the project. 
