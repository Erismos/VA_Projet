# TODO - Points bloquants restants

Date: 2026-04-21
Portée: ne garder ici que ce qui bloque encore la validation finale.

## 1) Validation bout-en-bout sur données réelles

- [ ] Exécuter un run complet P2 -> P3 -> P5 -> P4 sur au moins une séquence MOT17 réelle.
- [ ] Archiver les sorties de référence (rapports + fichiers générés) pour preuve de reproductibilité.
- [ ] Exécuter la commande `python -m project.cli e2e-smoke` sur données réelles et valider les artefacts de sortie.

## 2) Simplifications encore ouvertes

1. Simplifier P4 tracking
- [ ] Isoler les backends de détection dans des fonctions dédiées pour réduire la logique conditionnelle dans la boucle principale.
- [ ] Extraire la sérialisation MOT/summary dans des helpers réutilisables.

## 3) Critère de clôture

- [ ] Le run bout-en-bout est reproductible sur données réelles avec un seul guide d'exécution.