# GPU runbook: cum rulezi antrenarea pe Kaggle sau Colab

Antrenarea modelelor mari (RoBERT-large, XLM-R-large) se face pe GPU gratuit. Pe CPU nu este practică (măsurat local: ~47 s pentru un singur pas chiar și cu modelul mic RoBERT-small, D-022).

Ce primești la final: `submission.csv` (gata de încărcat pe platformă) și câte un zip per experiment cu probabilitățile OOF și de test.

---

## Pasul 0: construiește pachetul (pe calculatorul tău)

Din folderul repo-ului:

```
.venv\Scripts\python.exe scripts\package_src.py        # Windows
.venv/bin/python scripts/package_src.py                # Linux/Mac
```

Rezultat: `dist/polero-src.zip` (~1 MB: cod, configurații, notebook-uri, cele 3 CSV-uri și `artifacts/folds.csv`).

---

## Varianta A: Kaggle (recomandat, 2 GPU-uri T4)

1. **Încarcă pachetul ca dataset:** kaggle.com → *Datasets* → *New Dataset* → trage `polero-src.zip` → titlu `polero-src` → *Private* → *Create*. Kaggle dezarhivează automat zip-ul.
2. **Creează notebook-ul:** *Code* → *New Notebook* → *File* → *Import Notebook* → alege `notebooks/kaggle_train.ipynb`.
3. **Setări (panoul din dreapta, *Session options*):**
   - **Accelerator: `GPU T4 x2`** (două plăci de 16 GB; se antrenează două fold-uri în paralel). `GPU P100` merge și el, dar e un singur GPU și e mai lent.
   - **Internet: On** (necesar pentru descărcarea modelelor publice; contul Kaggle trebuie să fie verificat cu telefonul).
   - **Persistence:** nu e necesar.
4. **Adaugă datele:** *Add Input* → *Your Datasets* → `polero-src`.
5. **(Opțional, 2 minute) verificare:** în celula de setări pune `SMOKE = True`, rulează toate celulele interactiv (*Run All*). Trebuie să vezi `GPU 0: Tesla T4` și la final `exit code 0`. Apoi pune înapoi `SMOKE = False`.
6. **Rularea reală:** *Save Version* → *Save & Run All (Commit)* → *Save*. Rularea continuă în fundal (maximum 12 ore); poți închide browserul.
7. **Descarcă rezultatele:** deschide versiunea salvată → tab-ul *Output* → descarcă `out/submission.csv` și `out/*.zip`.

Precizia și mărimea batch-ului se aleg automat: fp16 pe T4/P100, bf16 pe L4/A100; numărul de tokeni per pas depinde de memoria plăcii.

---

## Varianta B: Google Colab (1 GPU T4 gratuit)

1. Pune `polero-src.zip` în rădăcina Google Drive (`My Drive/polero-src.zip`).
2. colab.research.google.com → *File* → *Upload notebook* → `notebooks/colab_train.ipynb`.
3. **Runtime → Change runtime type → Hardware accelerator: `T4 GPU`** → *Save*. (Cu Colab Pro poți alege `L4` sau `A100`; totul se adaptează automat și merge mai repede.)
4. **Runtime → Run all**, apoi permite accesul la Google Drive.
5. Rezultatele se salvează în Drive după fiecare fold (`My Drive/polero_artifacts`). **Dacă Colab se deconectează, dai din nou *Run all*: fold-urile terminate sunt sărite.**
6. La final: `My Drive/polero_out/submission.csv` și zip-urile experimentelor.

Pe Colab gratuit ai un singur GPU și sesiuni limitate, deci cele trei experimente pot avea nevoie de 2–3 sesiuni. Ordinea din `CONFIGS` este ordinea de prioritate (`robert-large_base` primul).

---

## Ce rulează

| config | model | ce adaugă |
|---|---|---|
| `robert-large_base` | readerbench/RoBERT-large | modelul principal (multi-head, ferestre glisante, max-pool) |
| `robert-large_ctx` | readerbench/RoBERT-large | + întrebarea anterioară din interviu + președinte/tip context/an |
| `xlmr-large_base` | FacebookAI/xlm-roberta-large | backbone diferit, pentru diversitate în ansamblu |

Fiecare experiment: 7 fold-uri, max 15 epoci cu early stopping (patience 3) pe evasion F1 + clarity F1 pe fold-ul de validare. La final, `polero.predict` face media celor trei experimente, compară decodorii și ajustarea bias-ului pe OOF (nested CV) și scrie `submission.csv` cu clarity consecvent cu evasion.

**Durată:** nu am putut măsura încă pe GPU. Ca ordin de mărime, un experiment large pe 7 fold-uri ar trebui să ia câteva ore pe o placă T4. Timpul real per epocă apare în `train.log`.

---

## După rulare: adu rezultatele înapoi

Dezarhivează fiecare `<exp>.zip` în `artifacts/` din repo (ex.: `artifacts/robert-large_base/oof.csv`), apoi:

```
.venv\Scripts\python.exe -m polero.evaluate artifacts\robert-large_base\oof.csv
.venv\Scripts\python.exe -m polero.predict --exp robert-large_base --exp robert-large_ctx --exp xlmr-large_base --out submissions\sub_v1.csv
```

Primul scrie metricile OOF în `docs/RESULTS.md`, al doilea reface submisia local, cu validare.

## Probleme frecvente

- **`No GPU found`:** acceleratorul nu e activat (pasul 3).
- **Eroare la descărcarea modelului / `ConnectionError`:** pe Kaggle, *Internet* e oprit sau contul nu e verificat cu telefonul.
- **`CUDA out of memory`:** în config pune `"token_budget": 1024` (în loc de `"auto"`) și rulează din nou; fold-urile terminate sunt sărite.
- **Kaggle s-a oprit la 12 ore:** rulează o versiune nouă doar cu experimentele rămase în `CONFIGS`; zip-urile celor terminate sunt deja în *Output*.
