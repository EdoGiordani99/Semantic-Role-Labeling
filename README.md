
# Semantic Role Labeling using Contextualized Word Embeddings and Attention Layers

Semantic Role Labeling is a taks that consists in assigning to each word in an input sentence labels that indicates their semantic role (such as agent, goal, or result)*. 

When we read a sentence, we are able to identify subject, object and other arguments mainly by looking to the predicate. In the example **“The cat eats the mouse”**, “the cat” is the agent while “the mouse” is the patient. It is therefore sufficient to change the verbal form of the predicate to make the roles reverse. In the sentence  "The mouse is eaten by the cat”, even if the meaning is the same, roles of the two arguments are reversed. Predicates play one of the most important roles in the SRL task. We can picture the prediction process as a pipeline of four steps:

<p align="center">
  <img src="images/srl_pipeline.png" width="460" height="250">
</p>


In this work I will only focus on the last 2 tasks.

## Dataset
The dataset I used to train my model is the UniteD-SRL (Tripodi et al., EMNLP 2021). This dataset is a private one and was provided by the [Sapienza NLP Group](https://github.com/SapienzaNLP). The dataset is a JSON file where each entry(each sample) is a dictionary containing the following fields:

```python
sentence_id: {
    “words”: [“The”, “cat”, “ate”, “the”, “mouse”, “.”],
    “lemmas”: [“the”, “cat”, “eat”, “the”, “mouse", “.”],
    “pos_tags”: [“DET”, ..., “PUNCT”],
    “dependency_relations”: [“NMOD”, ..., “ROOT”, ..., “P”],
    “dependency_heads”: [1, 2, 0, ...],
    “predicates”: [“_”, “_”, “EAT_BITE”, “_”, “_”, “_”],
    “roles”: {
        “2”: [“_”, “Agent”, “_”, “_”, “Patient”, “_”],
    }
}
```

* Words: list of words used in the sentence
* Lemmas: list of the base form under which the word is entered
* Dependency Head: syntattic information
* 
* Predicate: list where each predicate has it's own meaning (predicate disambiguation)
* PoS tags: list of the part of speech (such as verb, punctuation, noun, etc...)
* SRL tags: list of the semantic role of each word / token


