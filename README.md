# Knowledge Distillation zur Alterserkennung

---


## Installation

### Repository klonen

```bash id="j8s2kl"
git clone https://github.com/Cuzimnero/Praktikum-KI-Sicherheit
cd Praktikum-KI-Sicherheit
```

### Conda-Umgebung erstellen

```bash id="8dzv6r"
conda create --name env_kd python=3.14.4
```

### Conda-Umgebung aktivieren

```bash id="2mld6v"
conda activate env_kd
```

### Abhängigkeiten installieren

```bash id="7tnl4x"
pip install -r requirements.txt
```
### Mivolo herunterladen
```bash id="j8s2kl"
cd models
git clone https://github.com/wildchlamydia/mivolo
```
### Umgebung initalisieren
- dataset.py ausführen um den Datensatz herunterzuladen
- processed_dataset.py ausführen um die in der Dokumentation genutzten Datensatz Varianten zu erstellen

---

## Datensatz

### Verzeichnisstruktur
Hinweis: Jede Überkategorie enthält immer eine Version mit Gruppen sowie ohne (Default)
```text id="h82msv"
data/
├── processed/
│   ├── default/
│   │   ├── default/
│   │   └── groups/
│   ├── k_fold/
│   │   ├── default/
│   │   └── groups/
│   └── scaled/
│       └── k_fold/
└── raw/
```

---

## Modelle und Checkpoints

### Teacher-Modell

```text id="p9w5tr"
models/MiVOLO/checkpoints/
```

### Student-Modell
```text id="p9w5tr"
models/runs
```
---

## Konfiguration
Konfigurationsdatei in:
```text id="p9w5tr"
config/config.yaml
```

### Allgemeine Trainingsparameter

| Parameter       | Beschreibung |
| --------------- | ------------ |
| `batch_size`    |  `Größe eines Trainingsbatches (Anzahl an Bildern die gleichzeitig dem Modell übergeben werden bevor einmal die Gewichte aktualisiert werden)`      |
| `learning_rate` |     `Wie stark werden bei deinem Schritt die Hyperparameter angepasst?`         |
| `yolo_num_epochs`        | `Anzahl der Trainingsepochen des Yolo Modells`             |
| `classes_count` |    `Anzahl der Klassen des Datensatzes`          |
| `num_data_loader_worker` |       `Wie viele parallele Prozesse laden und vorbereiten Daten für das Training?`       |

### Knowledge-Distillation-Parameter

| Parameter     | Beschreibung |
| ------------- | ------------ |
| `alpha`       |              |
| `beta`        |              |
| `temperature` |              |
| `sigma`       |              |
| `cs_threshold`       |              |
| `destillation_num_epochs`       |              |

---

## Trainingsläufe
Hinweis: Alle Trainingsläufe nutzen dieselbe Konfigurationsdatei. Ausführen der jeweiligen Datei führt einen Lauf ohne KD sowie mit KD und speichert die Ergebnisse automatisch.
```text id="p9w5tr"
notebooks/
```
---



### Metriken

Siehe Dokumentation

---

## Ergebnisse

### Speicherort
---
Hinweis: Der Default Datensatz hat über 90 Klassen dies macht die Confusion Matrix unlesbar, daher wird diese beim Training nicht erzeugt. Die Ergebnisse werden sowohl in der Konsole ausgegeben als auch in der Logdatei gespeichert.

#### Confusion Matrizen:
---
```text id="a2x7ks"
/results
```
#### Trainingslogs:
```text id="a2x7ks"
/logs
```
Hinweis: Arten von im Log gespeicherten Daten lassen sich in der Konfigurationsdatei festlegen.

---
