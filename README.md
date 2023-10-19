# Fact Verification based on Knowledge Base

## Table of Contents
- [Models Description](#model_description)
- [Launching the Service](#launching_the_service)
- [Mapping of Volumes](#mapping_of_volumes)
- [Models Metrics](#models_metrics)
- [Usage Examples](#usage_examples)

## Models Description
Fact Verification is the process of verifying the factual accuracy of questioned reporting and statements. This service checks the truthfulness of a statement against DBpedia (2015-10 version). It consists of two components: one for extracting structured knowledge from text in the form of (subject, predicate, object) and another for verifying the extracted triplets against the KB in a rule-based manner using SPARQL queries.

The triplet extraction component is powered by a generative model based on Bart base. To ensure that the triplets are generated with grounding in the KB, constrained beam search is utilized.

The service takes a statement as input and provides the label (True/False), the top three triplets, and the corrected sentence if the label is False and there is enough information in the statement to correct it.

## Launching the Service
To launch the service with Docker, use the following command:

```shell
docker-compose up --build fact-checking
```

## Mapping of Volumes
In the `docker-compose.yml` file, the default mapping of the volume with model checkpoints and databases is as follows:

`~/.deeppavlov:/root/.deeppavlov`

You can customize this mapping by changing it to your preferred local directory:

`<your_custom_local_directory>:/root/.deeppavlov`

## Models Metrics
The model was trained and tested on [FactKG](https://arxiv.org/abs/2305.06590) and achieved the following metrics:

- Binary Classifier F1 = 70.0
- Binary Classifier Accuracy = 75.12
- Triplet Extraction F1 = 68.65

## Usage Examples
To interact with the service, you can use Python requests. Here's an example:

```python
import requests

res = requests.post("http://0.0.0.0:8008/respond", 
       json={"claims": ["Barack Obama was born in Kazakhstan."]}).json()

print(res)
```

This will return:
```python
[{
    'claim': 'Barack Obama was born in Kazakhstan.', 
    'corrected_claim': 'Barack Obama was born in Hawaii.',
    'pred': False,
    'triplets_topk': [[['Barack Obama', 'birthPlace', 'Kazakhstan']], [['Barack Obama', 'placeOfBirth', 'Kazakhstan']]
}]
```

### Get Metrics on FactKG dataset
To obtain metrics on the FactKG dataset, use the following code:

```python
import requests

res = requests.post("http://0.0.0.0:8008/get_metrics", 
       json={"num_samples": 100, "claim_type": "all"}).json()

print(res)
```

This will give you accuracy and F1 score metrics.
