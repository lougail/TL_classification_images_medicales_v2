# Classification de Pneumonie sur Radiographies Thoraciques avec Transfer Learning & MLflow

## 🩺 Contexte du projet

Ce projet vise à classifier automatiquement des radiographies thoraciques pour détecter la présence ou l’absence de pneumonie, en utilisant le deep learning et le transfer learning. L’objectif est d’assister les professionnels de santé avec un outil d’aide au diagnostic rapide et fiable.

Deux architectures pré-entraînées sont comparées :
- **MobileNetV2** (modèle léger et rapide)
- **ResNet50** (modèle plus profond et puissant)

Le suivi des expériences et des résultats est assuré avec **MLflow**.

---

## ⚙️ Installation des dépendances

Installe les dépendances nécessaires avec :

```sh
pip install -r requirements.txt
```

Principaux packages utilisés :
- tensorflow
- scikit-learn
- matplotlib
- seaborn
- mlflow

---

## 📂 Données

Le dossier `data/chest_xray/` doit contenir les sous-dossiers `train/`, `val/`, `test/` avec les images classées par dossier de classe.

**⚠️ Les données ne sont pas incluses dans ce dépôt.**  
Merci de télécharger le dataset (par exemple [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) et de le placer dans `data/chest_xray/`.

Un fichier `.gitkeep` est présent pour versionner le dossier même vide.

---

## 🚀 Lancer le notebook

Ouvre le notebook dans Jupyter ou VS Code :

```sh
jupyter notebook classification_pneumonie_vit.ipynb
```
ou
```sh
code classification_pneumonie_vit.ipynb
```

Exécute les cellules dans l’ordre pour entraîner et comparer les modèles.

---

## 📊 Suivi des expériences avec MLflow

Pour visualiser les runs et les métriques :

1. Lance le serveur MLflow dans le terminal :
   ```sh
   mlflow ui
   ```
2. Ouvre [http://localhost:5000](http://localhost:5000) dans ton navigateur.

Tu pourras comparer les performances, télécharger les modèles, et visualiser les artefacts (courbes, matrices de confusion…).

---

## 📝 Résultats et observations

- **MobileNetV2** : rapide à entraîner, bonnes performances sur des jeux de taille modérée.
- **ResNet50** : meilleure capacité à généraliser sur des motifs complexes, potentiellement plus précis mais plus lent.

Les métriques principales (accuracy, f1-score, matrice de confusion) sont loguées dans MLflow et sauvegardées dans le dépôt (`classification_report.txt`, `confusion_matrix.png`, `learning_curves.png`).

**Comparaison** :
- Si MobileNetV2 offre des performances proches de ResNet50, il est à privilégier pour la rapidité et la simplicité de déploiement.
- Si ResNet50 surpasse nettement MobileNetV2, il sera retenu pour des contextes où la précision prime.

---

## 📌 Pour aller plus loin

- Tester d’autres architectures (EfficientNet, DenseNet…)
- Ajouter de la data augmentation
- Adapter le pipeline à d’autres pathologies ou modalités d’imagerie

---

## Auteur

Projet réalisé dans le cadre d’un exercice de classification d’images médicales avec transfer learning et MLflow.