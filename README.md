# Projet S6

## Structure du Projet

Le projet est organisé en plusieurs dossiers et fichiers, chacun ayant un rôle spécifique. Voici une vue d'ensemble de la structure du projet et de son contenu :

### Dossiers

#### data
Ce dossier contient les données réelles sous forme de séries temporelles sur l'évolution des photons gamma.

#### generate_forms
Ce répertoire contient différents fichiers Python, chacun implémentant une génération de familles de formes spécifiques :

- **CR-DL** : Croissance rapide puis décroissance lente.
- **CL-DR** : Croissance lente puis décroissance rapide.
- **C** : Forme en cloche.
- **M** : Forme en "M".
- **P** : Pic assez serré.

#### matlab
Ce répertoire contient la démarche et l'extraction de familles de formes à partir des données réelles.

### Fichiers Python

#### create_dataset.py
Ce fichier contient un ensemble de fonctions qui permettent de générer des données (formes) selon nos souhaits :

- Définir le nombre de formes.
- Ajouter du bruit.
- Normaliser les données.
- Organiser les données dans des DataFrames pour qu'elles soient directement exploitables par nos algorithmes.

#### plot_graphs.py
Ce fichier contient un ensemble de fonctions pour :

- Dessiner nos formes.
- Dessiner nos métriques d'évaluation selon différents paramètres.
- Dessiner les matrices de confusion.
- Dessiner les formes classifiées pour une validation visuelle.

### Notebooks

#### motif_classification.ipynb
Ce notebook résume notre projet et notre démarche. Il inclut une analyse complète :

- Génération des données.
- Implémentation de l'algorithme ukmeans.
- Application des algorithmes kmeans et ukmeans.
- Évaluation de nos modèles avec différentes métriques d'évaluation.
- Analyse de l'influence du bruit.
- Analyse de l'influence du nombre de clusters.
- Analyse simultanée de l'influence du bruit et du nombre de clusters.
- Comparaison entre les deux méthodes.
- Observation du nombre de clusters trouvés par ukmeans et des facteurs qui peuvent influencer ce nombre.

