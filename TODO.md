# TODO - Points Bloquants Restants + Simplifications

Date: 2026-04-21
Portée: conserver uniquement les actions encore nécessaires avant validation finale.

## 1) Points bloquants restants

1. Validation bout-en-bout sur données réelles
- [ ] Exécuter un run complet P2 -> P3 -> P5 -> P4 sur au moins une séquence MOT17 réelle.
- [ ] Archiver les sorties de référence (rapports + fichiers générés) pour preuve de reproductibilité.
- [ ] Exécuter la commande `python -m project.cli e2e-smoke` sur données réelles et valider les artefacts de sortie.

2. Consolidation métriques projet
- [x] Produire un tableau unique consolidé (détection + tracking + performances) dans un artefact unique sous results.
- [x] Vérifier la cohérence des noms de modèles et colonnes de métriques entre P2, P3, P4, P5.

3. Robustesse opérationnelle
- [x] Ajouter un smoke test de non-régression (commande unique) couvrant au minimum conversion prédictions + tracking detections-json + génération rapport P5.
- [x] Valider explicitement les cas limites frame_id/classes sur un petit jeu de tests dédiés.

## 2) Simplifications / refactors proposés

1. Réduire la complexité de main_p5
- [x] Extraire la préparation data et l'évaluation dans deux fonctions orchestrables séparément.
- [x] Centraliser les chemins par défaut dans une petite config (dataclass ou YAML) pour éviter la duplication.

2. Nettoyer les modes de secours non essentiels
- [x] Limiter l'usage du mode mock aux tests/demo et le sortir du flux principal de prod.
- [x] Uniformiser la stratégie de fallback (messages d'erreur + arrêt explicite) au lieu de mélanger fallback implicite et exceptions.
- [x] Supprimer du code les branches mock/dummy fallback du pipeline P5.

3. Unifier les validateurs
- [x] Regrouper les validateurs de schéma et dataset dans un module unique de validation pour éviter les règles dispersées.
- [x] Harmoniser les structures de retour des validateurs (ok, reason, details) pour simplifier le chaînage.

4. Simplifier P4 tracking
- [ ] Isoler les backends de détection dans des fonctions dédiées pour réduire la logique conditionnelle dans la boucle principale.
- [ ] Extraire la sérialisation MOT/summary dans des helpers réutilisables.

5. Rationaliser la CLI projet
- [x] Ajouter une commande pipeline end-to-end dans project.cli pour éviter les enchaînements manuels.
- [x] Aligner les noms d'arguments similaires entre composants (output, output-dir, pred-file, gt-json).

## 3) Critère de clôture du TODO

- [ ] Le run bout-en-bout est reproductible sur données réelles avec un seul guide d'exécution.
- [x] Le tableau consolidé des métriques est généré automatiquement.
- [x] Les simplifications prioritaires (main_p5 + validateurs + commande pipeline) sont appliquées.

## 4) Nouveaux points d'attention après simplification

- [x] Ajouter 1 à 2 tests unitaires dédiés à `p5/validation.py` pour verrouiller le format normalisé (`ok`, `reason`, `details`).
- [x] Vérifier sur environnement cible que `project/e2e.py` couvre bien le cas sans P4 (`--skip-p4`) et le cas complet avec P4.
- [x] Vérifier que toutes les commandes d'exemple P5 pointent désormais vers des données réelles et non des chemins de secours.