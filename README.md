# Python_project
***
Groupe : BRINGER Erwan, CANTIE Maxime

## DATASET
Ce projet consiste en l’analyse d’un jeu de donnée recueillant des données sur la consommation de drogue, la personnalité des participants, ainsi que d’autres données les définissant :
https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
* Le dataset est composé de 1885 individus et 32 attributs sont étudiés pour chacun. Ainsi, on connait des données personnelles sur les participants (tel que la nationalité, l’Age, l’éducation, la personnalité, …) mais aussi s’ils ont déjà consommé une des drogues listées et si oui quand.

#### Attributs personnels :
 - Age
 - Gender
 - Education
 - Country
 - Ethnicity
 - Nscore (Neuroticism)
 - Escore (Extraversion)
 - Oscore (Openness to experience)
 - Ascore (Agreeableness)
 - Cscore (Conscientiousness) 
 - Impulsive (impulsiveness measured by BIS-11)
 - SS (sensation seeing measured by ImpSS)

#### Attributs drogue :
 
 - Alcohol     
 - Amphet 
 - Amyl
 - Benzos 
 - Caff
 - Cannabis
 - Choc
 - Coke 
 - Crack
 - Ecstasy
 - Heroin
 - Ketamine
 - Legalh
 - LSD
 - Meth 
 - Mushrooms
 - Nicotine 
 - VSA
 - Semer 

### Analyse du dataset:
La grande partie des participants se trouve dans la tranche d’âge 18-54
Les participants sont majoritairement originaires des USA ou du Royaume-Unis (UK)
Il y a un nombre d’homme et de femme équivalent
L’Ethnie majoritairement représenté est les blancs

## OBJECTIF 
Création d’un model permettant de déterminer et de prédire les risques de consommer une ou des drogues en fonction de sa personnalité et de sa situation sociale.
Après avoir analysé le dataset afin de décortiquer ses spécificités, nous avons choisi de tester plusieurs modèles afin de trouver le plus optimal : Logistic regression, up-sampling, XGBOOST, MLP, RF. Le modèle utilisé dépend pour chaque drogue et nous ne gardons que celui avec l’accuracy la plus élevée.
