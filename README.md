# MoneyTime

Ce projet est notre projet de fin de troisième années. MoneyTime est une IA qui a pour but de prédire l'évolution du prix des cryptos monnaies.

## Prérequis

Python >= 3.10, Git

## Setup du Projet

Téléchargement initial du projet
```bash
$ git clone https://github.com/Yann-Fournier/MoneyTime
```

Placer ensuite le terminal à l'intérieur du dossier MoneyTime 

Mise en place de l'environnement virtuel (très recommandé):
```bash
$ python -m venv .venv  
$ .venv\Scripts\activate  
$ pip install -r .\requirements.txt
```

## Téléchargement et formattage des données

- Créez un dossier nommé database à la racine du projet.
- Exécuter le script *utilities/Get_data.ipynb* pour télécharger l'historique des crypto-monnaies depuis Binance.
- Exécuter le script *utilities/Ajout_ind_bdd.ipynb* pour ajouter les indicateurs et les métriques utiles pour entrainer l'IA.

## NN

Neural Profet de FaceBook

LSTM + de l'attention + Convolution




