# Mémoire M1 – Analyse en Composantes Principales à Noyaux

Ce dépôt regroupe les fichiers liés à notre mémoire de Master 1 en mathématiques appliquées à l’Université Paris Dauphine - PSL. Le travail porte sur l’Analyse en Composantes Principales à noyaux (Kernel PCA), une méthode de réduction de dimension non linéaire, et ses applications concrètes.

📄 Le mémoire complet est disponible dans le fichier PDF :  
**`Mémoire M1 Kernel PCA - Kevin Wardakhan - Ibrahim Youssouf Abdelatif - Erwan Ouabdesselam.pdf`**

---

## 🎯 Présentation générale

L’objectif du mémoire est de présenter le cadre théorique de l’ACP à noyaux et de l’appliquer à des jeux de données où des relations non linéaires sont présentes. Chaque expérimentation met en avant les limites de l’ACP classique et les apports du noyau.

---

## 📚 Sommaire du mémoire

1. **Introduction**  
2. **Analyse en Composantes Principales (ACP)**  
   2.1 Définitions et notations  
   2.2 Théorème ACP  
   2.3 Problème d’optimisation  
   2.4 Résolution du problème  
3. **ACP à noyaux (Kernel PCA)**  
   3.1 Introduction et définitions  
   3.2 Théorème d’Aronszajn  
   3.3 Principe général  
   3.4 Algorithme  
   3.5 Exemples de noyaux  
   3.6 Conclusion théorique  
   3.7 ACP à noyaux incrémental  
4. **Étude 1 – Classification de sentiments IMDb**  
   4.1 Objectifs  
   4.2 Prétraitement du texte  
   4.3 Vectorisation  
   4.4 Réduction de dimension  
   4.5 Conclusion  
5. **Étude 2 – Détection d’anomalies sur MNIST**  
   5.1 Prétraitement  
   5.2 Application de l’ACP à noyaux  
   5.3 Erreur de reconstruction  
   5.4 Définition du seuil  
   5.5 Comparaison avec ACP  
   5.6 Paramètres optimaux  
6. **Étude 3 – Débruitage de signaux ECG**  
   6.1 Prétraitement des signaux  
   6.2 Ajout de bruit  
   6.3 Débruitage par projection-reconstruction
---

## 🧪 Détails des cas pratiques

### 1. Classification de sentiments – IMDb  
**Objectif** : évaluer si Kernel PCA permet d’améliorer les performances de modèles de classification sur des données textuelles.  
**Méthode** : les critiques IMDb sont nettoyées et vectorisées à l’aide d’un sac de mots (`CountVectorizer`). Une réduction de dimension est ensuite appliquée via Kernel PCA (noyau cosinus). Plusieurs modèles (régression logistique, SVM, KNN) sont entraînés sur les données réduites.  
Les performances sont comparées à celles obtenues sans réduction, ainsi qu’avec une ACP linéaire.

---

### 2. Détection d’anomalies – MNIST  
**Objectif** : détecter automatiquement, sans supervision, les chiffres qui ne font pas partie d’une classe connue.  
**Méthode** : le modèle est entraîné uniquement sur des images du chiffre "0", et toutes les images de test sont projetées dans l’espace de Kernel PCA (noyau RBF). L’**erreur de reconstruction** est utilisée pour distinguer les images similaires au "0" (normales) des autres (anormales).  
Un **seuil optimal** est déterminé pour la détection, et les résultats sont comparés à ceux obtenus avec une ACP classique.

---

### 3. Débruitage de signaux ECG  
**Objectif** : atténuer un bruit artificiel ajouté à des signaux ECG tout en conservant la structure physiologique du signal.  
**Méthode** : des segments de signaux propres sont extraits du MIT-BIH Arrhythmia Dataset, puis perturbés par un bruit gaussien. Kernel PCA (noyau RBF) est utilisé pour apprendre une représentation non linéaire à partir des signaux propres. Les signaux bruités sont ensuite projetés et reconstruits dans cet espace.  
La qualité du débruitage est évaluée par comparaison avec l’ACP classique, notamment via l’erreur quadratique moyenne (MSE).

---

## 👨‍🏫 Auteurs

Mémoire réalisé dans le cadre du Master 1 à Dauphine, sous la direction de :
- M. Denis Pasquignon  
- M. Patrice Bertrand

Par :
- Erwan Ouabdesselam  
- Ibrahim Youssouf Abdelatif  
- Kevin Wardakhan

