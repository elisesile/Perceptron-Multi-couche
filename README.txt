
**************************************************
**************          IA+         **************
**************  TP1: Autoencodeur   **************
**************      Elise NOGA      **************
**************************************************

(TP effectué lors du M1 IISC à l'UCP - 2h de temps)

Fichier mlpMultFiles.py :

Le perceptron multi-couches fourni et modifié de manière à :
- Ce que les entrées d'entrainement et de test soient des 
listes de 784 valeurs
- Ce que l'entraînement se fasse sur la totalité de la base
- Ce que les tests s'effectuent sur 30 images tirées au hasard
dans la base de test
- Ce que les erreurs successives soient affichées à la fin de 
l'entraînement
- Ce que les images prédites et réelles soient affichées après 
chaque test

Fichier mlpCsv.py :

Même implémentation que le précédent mais récupération 
des données pour l'entraînement et le test à partir des
fichiers csv.

Test 1 et Test 2 :

Dossiers qui contiennent les résultats de deux itérations. 
Pour chacune, l'algorithme a été entraîné avec l'ensemble de
la base.
Les évolutions de l'erreurs sont incluses.

![alt text]https://github.com/elisesile/Perceptron-Multi-couche/blob/main/image01input.png


Difficultées rencontrées : 

Après plusieurs essais, je n'ai pas réussi à faire rentrer 
l'activation de la couche centrale dans le calcul de l'erreur.
