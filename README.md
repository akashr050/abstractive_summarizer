# Abstractive Summarization and Extrinsic Evaluation via Q/A
Text summarization is a process to create a representative summary or abstract of the entire document, by finding the most informative portions of the article. There are two approaches for automatic summarization: extractive summarization and abstractive summarization. The current techniques to evaluate an summarizer are BLEU and Rogue-n scores. These metrics are based on the overlap between the predicted summaries and the summaries provided by human (generally mechanical turks or news headlines). These metrics can be good system to evaluate the extractive summaries because they extract word features from the input text. Hence, we expect there to be a huge overlap between the predicted and human-provided summaries. For abstractive summarizer which aims to understand the text and provide a summary, it is not necessary for them to have the same words as there are in the human-provided summaries. But due to non-availability of a better metric system, we are still using BLEU and Rogue-n scores to evaluate abstractive summaries. Our understanding, is that if a summary can answer the questions based on the text then it is a good summary. Hence, we propose to use the Question/Answering system as an evaluation metric to evaluate the summaries. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
