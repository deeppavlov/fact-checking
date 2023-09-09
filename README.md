# Fact Verification based on KB 


<a name="model_description">Models description</a>
------------------

Fact Verification is the process of verifying the factual accuracy of questioned reporting and statements. The service in this repo checks the truthfullness of a statemnt against DBpedia (2015-10 version). It consists of a component for extracting structured knowledge from text in the form of (subject, predicate, object) and a component to verify the extracted triplets against the KB in a ruke-based manner, using SPARQL queries.

The triplet extraction component is a generative model based on Bart base. To ensure that the triplets are generated grounded on the KB, constrained beam serach is used.

The service requires a statement as input and outputs the label (True/False), top 3 triplets and the corrected sentence if the label is False and there is enough information in the statement to correct it. 

<a name="lauch_services">Launching the services</a>
----------------------

Lauch the service with docker :

```shell
docker-compose up --build fact-checking
```


<a name="volumes">Mapping of volumes</a>
------------------

In docker-compose.yml the default mapping of the volume with model checkpoints and databases in the following:

 ~/.deeppavlov:/root/.deeppavlov

You can change this mapping to your custom:

 <your_custom_local_directory>:/root/.deeppavlov



<a name="metrics">Models metrics</a>
--------------

The model was trained and tested on [FactKG](https://arxiv.org/abs/2305.06590) and achieved:

Binary Classifier F1 = 70.09
Binary Classifier Accuracy = 76.11
Triplet extraction F1 = 61.97
