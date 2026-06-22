# MeetingScribe – Dokumentation für jedermann

> **MeetingScribe** ist ein Python-Programm, das Meetings automatisch aufnimmt, in Text umwandelt, verschiedene Sprecher erkennt und am Ende professionelle Protokolle erstellt – wahlweise auch mit Hilfe einer KI.

---

## Inhaltsverzeichnis

1. [Was macht MeetingScribe überhaupt?](#was-macht-meetingscribe-überhaupt)
2. [Wie läuft das Programm ab? (Gesamtablauf)](#wie-läuft-das-programm-ab)
3. [Die wichtigsten Bausteine erklärt](#die-wichtigsten-bausteine-erklärt)
   - [Audio aufnehmen](#1-audio-aufnehmen)
   - [Sprache erkennen (Whisper)](#2-sprache-erkennen-whisper)
   - [Sprachsegmente herausschneiden (Segmentierung)](#3-sprachsegmente-herausschneiden-segmentierung)
   - [Sprecher unterscheiden (Diarisierung)](#4-sprecher-unterscheiden-diarisierung)
   - [Protokolle erstellen](#5-protokolle-erstellen)
   - [KI-Verbesserung (optional)](#6-ki-verbesserung-optional)
4. [Welche Dateien entstehen?](#welche-dateien-entstehen)
5. [Wie startet man das Programm?](#wie-startet-man-das-programm)
6. [Alle Einstellmöglichkeiten im Überblick](#alle-einstellmöglichkeiten-im-überblick)
7. [Technische Voraussetzungen](#technische-voraussetzungen)
8. [Häufige Fragen (FAQ)](#häufige-fragen-faq)

---

## Was macht MeetingScribe überhaupt?

Stell dir vor, du hast ein Meeting, ein Telefonat oder eine Videokonferenz. MeetingScribe läuft dabei still im Hintergrund und:

1. **Nimmt den Ton auf** – entweder vom Mikrofon oder direkt vom Computerausgang (also alles, was aus den Lautsprechern kommt).
2. **Erkennt automatisch, wann jemand spricht** – Stille wird ignoriert, nur echte Sprache wird verarbeitet.
3. **Wandelt das Gesprochene in Text um** – mithilfe von Whisper, einem der besten KI-Spracherkennungsmodelle der Welt (von OpenAI).
4. **Unterscheidet verschiedene Sprecher** – Das Programm versucht herauszufinden, wer wann gesprochen hat, und nennt sie „Sprecher_1", „Sprecher_2" usw.
5. **Erstellt am Ende Protokolle** – ein kurzes, ein mittleres und ein ausführliches Protokoll, wahlweise auch KI-optimiert.

---

## Wie läuft das Programm ab?

```
Start
  │
  ├─ Mikrofon oder Systemton? ──→ Audio wird in Echtzeit aufgenommen
  │
  ├─ Ist gerade jemand am Reden? ──→ Ja: Sprachsegment wird ausgeschnitten
  │
  ├─ Segment wird transkribiert (Whisper) ──→ Text entsteht
  │
  ├─ Wer hat gesprochen? (Stimmanalyse) ──→ Sprecher wird zugewiesen
  │
  ├─ Text + Sprecher werden gespeichert
  │
  └─ Am Ende: Protokolle werden erstellt (optional mit KI verfeinert)
```

Das alles passiert **gleichzeitig und in Echtzeit**, während das Meeting noch läuft.

---

## Die wichtigsten Bausteine erklärt

### 1. Audio aufnehmen

Das Programm kann auf zwei Arten Audio aufnehmen:

**Mikrofon-Modus (`--mic`)**
Nimmt auf, was das eingebaute oder angeschlossene Mikrofon aufnimmt – also alles im Raum.

**System-Loopback-Modus (Standard)**
Nimmt den Ton auf, der gerade aus dem Computer kommt – also z. B. die Stimmen in einem Zoom-Call oder Teams-Meeting. Dafür wird entweder WASAPI (Windows) oder PyAudioWPatch verwendet. Das ist besonders nützlich, weil der Ton dabei sehr klar ist.

> **Einfach erklärt:** Der Loopback-Modus ist so, als würde man ein Kabel direkt vom Lautsprecherausgang des Computers zurück in den Eingang stecken – man nimmt auf, was der Computer selbst abspielt.

---

### 2. Sprache erkennen (Whisper)

Das Herzstück der Texterkennung ist **Faster-Whisper**, eine optimierte Version des Whisper-Modells von OpenAI.

- Das Modell läuft auf der **Grafikkarte (GPU)**, wenn eine NVIDIA-Karte vorhanden ist – das ist viel schneller.
- Ist keine GPU vorhanden, läuft es auf der **CPU** – langsamer, aber funktioniert trotzdem.
- Das Programm erzwingt immer das Modell `large-v3`, weil das die beste Erkennungsqualität bietet.
- Die Sprache ist auf **Deutsch** eingestellt (`language="de"`).
- Ein **VAD-Filter** (Voice Activity Detection) sorgt dafür, dass nur echte Sprache und kein Rauschen transkribiert wird.

> **Einfach erklärt:** Whisper ist wie ein sehr guter menschlicher Stenograf, der zuhört und alles mitschreibt. Die „large-v3"-Version ist der erfahrenste Stenograf im Team.

---

### 3. Sprachsegmente herausschneiden (Segmentierung)

Nicht jede Millisekunde des Audios wird einzeln an Whisper geschickt – das wäre zu ineffizient. Stattdessen übernimmt der **EnergySegmenter** die Vorarbeit:

- Er misst ständig die **Lautstärke** des Tons (in Dezibel).
- Ist es zu leise → Stille, kein Sprechen.
- Wird es lauter → jemand fängt an zu reden → Segment beginnt.
- Wird es wieder leiser und bleibt leise → Segment endet.
- Das fertige Segment (ein Audioschnipsel von wenigen Sekunden) wird dann an Whisper übergeben.

**Wichtige Parameter:**
| Parameter | Bedeutung |
|---|---|
| `min_speech_seconds` (0,55 s) | Mindestlänge, damit ein Segment als Sprache gilt |
| `min_silence_seconds` (0,35 s) | Wie lange Stille sein muss, bevor ein Segment endet |
| `max_segment_seconds` (18 s) | Maximale Länge eines Segments |

> **Einfach erklärt:** Der Segmenter funktioniert wie ein Tonassistent, der nur dann das Aufnahmegerät einschaltet, wenn wirklich jemand redet – und es wieder ausschaltet, wenn es still wird.

---

### 4. Sprecher unterscheiden (Diarisierung)

Das ist der komplexeste Teil des Programms. Es versucht, aus der Stimme zu erkennen, **wer** gesprochen hat.

**Wie funktioniert das?**

Jede Stimme hat eine einzigartige „Klangfarbe" – Tonhöhe, Stimmfrequenz, Resonanz usw. Das Programm berechnet für jeden Sprachschnipsel einen sogenannten **Embedding-Vektor** – eine Liste von Zahlen, die den Klang der Stimme mathematisch beschreibt.

Dann wird verglichen:
- Ähnelt dieser Vektor dem einer bekannten Stimme? → Gleicher Sprecher
- Unterscheidet er sich stark? → Neuer Sprecher wird angelegt

**Zwei Methoden stehen zur Verfügung:**

1. **SpeechBrain (Standard, besser):** Ein spezialisiertes KI-Modell (`spkrec-ecapa-voxceleb`), das auf Millionen von Stimmproben trainiert wurde. Liefert präzise Stimmvektoren.

2. **Fallback (ohne KI-Modell):** Falls SpeechBrain nicht installiert ist, berechnet das Programm selbst grobe Stimmmerkmale wie Tonhöhe, Frequenzschwerpunkt und Nulldurchgangsrate. Weniger präzise, aber funktioniert überall.

**Bekannte Sprecherprofile:**
Man kann dem Programm vorab Stimmprofile von bekannten Personen mitgeben (`--speaker-profiles-in`). Dann wird versucht, diese Personen namentlich zu erkennen, statt sie anonym als „Sprecher_1" zu bezeichnen.

> **Wichtiger Hinweis aus dem Code:** Sprecherlabels können falsch sein. Die KI macht Fehler, besonders wenn viele Personen gleichzeitig reden oder die Audioqualität schlecht ist. Die Protokolle weisen ausdrücklich darauf hin.

---

### 5. Protokolle erstellen

Am Ende des Meetings werden automatisch **vier verschiedene Ausgaben** erzeugt:

#### Vollständiges Transkript (`meeting.txt`)
Eine chronologische Liste aller erkannten Sprachsegmente mit Zeitstempel, Sprecher und Text:
```
[00:01-00:08] Sprecher_1: Guten Morgen, können wir anfangen?
[00:09-00:23] Sprecher_2: Ja, ich bin soweit. Fangen wir mit Punkt 1 an.
```

#### Dialog-Bericht (`meeting_dialog.md`)
Dasselbe wie oben, aber als strukturiertes Markdown-Dokument mit einer zusätzlichen Ansicht nach Sprecher sortiert.

#### Ausführliches Protokoll (`meeting_report_full.md`)
Enthält die vollständige Timeline plus den gesamten Transkripttext am Stück.

#### Langprotokoll (`meeting_report_long.md`)
Eine verdichtete Version mit:
- Zusammenfassung
- Erkannte Kernthemen (häufig erwähnte Begriffe)
- Erkannte Entscheidungen (z. B. Sätze mit „wir machen", „wir nutzen")
- Offene Punkte / To-dos (z. B. Sätze mit „müssen", „prüfen", „abklären")
- Timeline

#### Kurzprotokoll (`meeting_protocol_short.md`)
Ultra-kurze Version: maximal 8 Bulletpoints, nur das Wesentlichste.

---

### 6. KI-Verbesserung (optional)

Mit dem Parameter `--llm-finalize` kann man nach der Aufnahme eine KI (z. B. GPT-4 oder ein lokales Ollama-Modell) bitten, die Protokolle zu verbessern.

**Was macht die KI dann?**
- Sie bekommt das gesamte Rohmaterial (alle Segmente, Zeitstempel, Sprecher)
- Sie erstellt daraus sauber formulierte, thematisch gegliederte Protokolle auf Deutsch
- Sie extrahiert Entscheidungen und To-dos zuverlässiger als die einfachen Textalgorithmen

**Zwei Modi:**

1. **Direktmodus** (kurze Meetings): Alles wird auf einmal an die KI geschickt.
2. **Chunk-Modus** (lange Meetings): Das Meeting wird in Abschnitte aufgeteilt, jeder Abschnitt wird zuerst zusammengefasst, dann werden alle Zusammenfassungen zu einem Gesamtprotokoll kombiniert.

**Kompatible KI-Dienste:**
- OpenAI (GPT-4, GPT-4o, etc.)
- Jedes OpenAI-kompatible API – auch lokale Modelle via [Ollama](https://ollama.com) (`http://localhost:11434/v1`)

> **Einfach erklärt:** Stell dir vor, du gibst einem sehr guten Assistenten alle Notizen und er schreibt daraus ein professionelles Protokoll. Genau das macht die KI hier.

---

## Welche Dateien entstehen?

Nach einer Aufnahme findet man im Sitzungsordner (z. B. `meetings/meeting_20250427_143022/`) folgende Dateien:

| Datei | Inhalt |
|---|---|
| `meeting.wav` | Die rohe Audioaufnahme |
| `meeting.txt` | Vollständiges Transkript als Text |
| `meeting_raw_segments.jsonl` | Jedes Segment als technische JSON-Zeile |
| `meeting_raw.json` | Alle Rohdaten strukturiert als JSON |
| `meeting_dialog.md` | Dialog-Bericht mit Sprecheransicht |
| `meeting_report_full.md` | Ausführliches Protokoll |
| `meeting_report_long.md` | Kompaktes Langprotokoll |
| `meeting_protocol_short.md` | Ultra-Kurzprotokoll |
| `meeting_speakers.jsonl` | Sprecher-Log |
| `speaker_profiles_auto.json` | Automatisch erlernte Stimmprofile |

---

## Wie startet man das Programm?

### Einfachster Start (Systemton, 10 Minuten)
```bash
python meetingscribe.py
```

### Mikrofon aufnehmen
```bash
python meetingscribe.py --mic
```

### Systemton aufnehmen (Windows WASAPI Loopback)
```bash
python meetingscribe.py --output-loopback
```

### 30-minütiges Meeting aufnehmen
```bash
python meetingscribe.py --duration 1800
```

### Mit KI-Protokoll (OpenAI)
```bash
python meetingscribe.py --llm-finalize --llm-api-key sk-... --llm-model gpt-4.1-mini
```

### Mit lokalem Ollama-Modell
```bash
python meetingscribe.py --llm-finalize --llm-api-base http://localhost:11434/v1 --llm-model llama3
```

### Verfügbare Audiogeräte anzeigen
```bash
python meetingscribe.py --list-devices
```

### Meeting in bestimmtem Ordner speichern
```bash
python meetingscribe.py --meeting-folder C:\Protokolle\MeinMeeting
```

Die WAV-Ausgabe wird als `PCM_16` gespeichert und ist dadurch kleiner als vorher.

---

## Alle Einstellmöglichkeiten im Überblick

### Aufnahme

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--duration` | 600 | Aufnahmedauer in Sekunden |
| `--mic` | aus | Mikrofon statt Systemton verwenden |
| `--output-loopback` | aus | WASAPI Loopback (Windows) verwenden |
| `--device` | - | Gerätename (Teiltext reicht) |
| `--device-index` | - | Genaue Gerätenummer (aus `--list-devices`) |
| `--no-save-audio` | aus | Kein WAV speichern |
| `--list-devices` | - | Geräteliste anzeigen und beenden |

### Spracherkennung

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--model` | large-v3 | Whisper-Modellgröße (immer large-v3) |
| `--min-speech-seconds` | 0.55 | Mindestlänge eines Sprachsegments |
| `--min-silence-seconds` | 0.35 | Stille bis Segment endet |
| `--max-segment-seconds` | 18.0 | Maximale Segmentlänge |

### Sprechererkennung

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--speaker-backend` | speechbrain | speechbrain oder fallback |
| `--max-speakers` | 8 | Maximale Anzahl erkannter Sprecher |
| `--speaker-distance-threshold` | 0.28 | Schwellenwert für „neue Stimme" (0–1) |
| `--speaker-profiles-in` | - | Bekannte Stimmprofile laden |
| `--speaker-profiles-out` | auto.json | Erlernte Profile speichern |
| `--cpu-only` | aus | GPU für SpeechBrain deaktivieren |

### Dateipfade

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--out` | meeting.wav | Audiodatei |
| `--txt` | meeting.txt | Transkript |
| `--storage-root` | meetings | Oberordner für Sitzungen |
| `--session-prefix` | meeting | Präfix des Sitzungsordners |
| `--meeting-folder` | - | Fester Ordnerpfad |

### KI-Protokollierung

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--llm-finalize` | aus | KI-Protokolle aktivieren |
| `--llm-api-base` | OpenAI | API-Adresse |
| `--llm-model` | gpt-4.1-mini | Modellname |
| `--llm-api-key` | - | API-Schlüssel |
| `--llm-timeout-seconds` | 600 | Timeout für KI-Anfragen |
| `--llm-max-input-chars` | 120000 | Max. Zeichen für direkten Modus |
| `--llm-chunk-input-chars` | 45000 | Max. Zeichen pro Chunk |
| `--llm-system-prompt` | - | Eigener Systembefehl für die KI |

---

## Technische Voraussetzungen

Folgende Python-Pakete müssen installiert sein:

| Paket | Wofür |
|---|---|
| `faster-whisper` | Spracherkennung |
| `sounddevice` | Audio aufnehmen (Mikrofon/WASAPI) |
| `soundfile` | WAV-Dateien lesen und schreiben |
| `numpy` | Zahlen- und Audioverarbeitung |
| `torch` | GPU-Unterstützung (optional) |
| `speechbrain` | Sprechererkennung (optional, aber empfohlen) |
| `pyaudiowpatch` | Windows Loopback-Aufnahme (optional) |

**Für GPU-Beschleunigung** wird eine NVIDIA-Grafikkarte mit CUDA-Unterstützung empfohlen. Das Programm fällt automatisch auf die CPU zurück, wenn keine GPU gefunden wird.

---

## Häufige Fragen (FAQ)

**Warum heißen die Sprecher „Sprecher_1", „Sprecher_2"?**
Das Programm erkennt Stimmen automatisch, kennt aber keine Namen. Es gruppiert ähnliche Stimmen zusammen. Wer genau welche Nummer ist, muss man nachher selbst zuordnen. Mit `--speaker-profiles-in` kann man bekannte Stimmen vorab einspeisen.

**Warum ist die Sprechererkennug manchmal falsch?**
Sprecherdiarisierung ist ein schwieriges Problem. Faktoren wie Hintergrundgeräusche, ähnliche Stimmen, gleichzeitiges Reden oder schlechte Audioqualität können die Erkennung verschlechtern. Das Programm weist in den Protokollen ausdrücklich darauf hin.

**Kann ich die KI-Protokolle ohne Internetverbindung nutzen?**
Ja! Mit einem lokal laufenden [Ollama](https://ollama.com)-Server (`--llm-api-base http://localhost:11434/v1`) funktioniert alles offline.

**Was passiert, wenn ich das Programm mit Strg+C abbricht?**
Das Programm erkennt die Unterbrechung sauber, verarbeitet noch alle bereits aufgenommenen Segmente und erstellt trotzdem vollständige Protokolle.

**Wie lange kann ein Meeting sein?**
Theoretisch unbegrenzt. Mit `--duration 86400` würde es 24 Stunden aufnehmen. Für sehr lange Meetings verwendet die KI-Verarbeitung automatisch den Chunk-Modus, um Speicherlimits zu umgehen.

**Welche Sprachen werden unterstützt?**
Im Code ist Deutsch (`language="de"`) fest eingestellt. Um andere Sprachen zu verwenden, müsste man den Code anpassen.

---

*Erstellt automatisch aus dem Quellcode von MeetingScribe.*
