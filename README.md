# ACP à noyaux – Codes du mémoire de Master 1

Ce repository contient l’ensemble des codes Python développés dans le cadre du mémoire de Master 1 intitulé *Analyse en Composantes Principales à noyaux*, soutenu à l’Université Paris Dauphine-PSL (2025).

##  Contenu du mémoire

Le mémoire explore l’extension non linéaire de l’Analyse en Composantes Principales (ACP) via les méthodes à noyaux, en détaillant :
- les fondements théoriques de l’ACP et de l’ACP à noyaux,
- la mise en œuvre algorithmique de la KPCA,
- plusieurs expérimentations sur données réelles.

##  Expériences incluses

### 1. Analyse de sentiments sur IMDb
- Prétraitement linguistique avancé (lemmatisation, POS-tagging, etc.)
- Vectorisation Bag-of-Words
- Réduction de dimension par ACP et ACP à noyaux (cosinus)
- Classification supervisée (LogReg, SVM, KNN)

### 2.  Détection d’anomalies sur MNIST
- Réduction des images (8x8)
- Entraînement KPCA sur les chiffres "0"
- Erreur de reconstruction comme score de nouveauté

### 3.  Débruitage de signaux ECG
- Utilisation des bases MIT-BIH et NSTDB
- Contamination contrôlée avec différents bruits (ma, em, wn)
- Reconstruction avec KPCA vs ACP


