
# Semantic Role Labeling
## using Contextualized Word Embeddings and Attention Layers

Semantic Role Labelling is a taks that consists in assigning to each word in an input sentence labels that indicates their semantic role (such as agent, goal, or result)*. 

When we read a sentence, we are able to identify subject, object and other arguments mainly by looking to the predicate. In the example “The cat eats the mouse”, “the cat” is the agent while “the mouse” is the patient. It is therefore sufficient to change the verbal form of the predicate to make the roles reverse. In the sentence  "The mouse is eaten by the cat”, even if the meaning is the same, roles of the two arguments are reversed. Predicates play one of the most important roles in the SRL task. We can picture the prediction process as a pipeline of four steps: 
1)	Identification of the predicate of the sentence
2)	Associate a meaning to the found predicate    
3)	Identifying the arguments of the predicate
4)	Classifying them with their classes
In my work I only focused on the last 2 points
