# Lettuce Care Pakistan (v1.3.2)

**Lettuce Care Pakistan** is a local, Streamlit-based application for classifying lettuce leaf images and producing context-aware crop-care guidance. It combines a selected **EfficientNet-B0 classifier trained with no random erasing and balanced classifier retraining (cRT)** with five deterministic software agents.

This release accompanies the study **“Two-Stage EfficientNet-B0 Classification with Contextual Care Guidance for Lettuce.”** It contains the selected application checkpoint and the implementation of the care-guidance workflow. It does **not** contain the entire training/ablation pipeline or the original image datasets.

## 1. Requirements

- **Python 3.12** (recommended for this packaged release)
- Windows, Linux, or macOS
- Internet access for live weather, place search, and online soil lookups; bundled Pakistan soil maps can provide some background information without live soil access
- A suitable Python environment with **PyTorch** and **torchvision**; the application can run on CPU or use a CUDA-capable GPU when available

The included `requirements.txt` declares Streamlit, NumPy, Pillow, PyTorch, torchvision, and rasterio. Installation of PyTorch can require a substantial download.

## 2. Download and run

### Step 1. Extract the release

Extract `Pakistan_Lettuce_DSS_v1.3.2_Clean.zip` and open a terminal **inside the extracted `pakistan_lettuce_dss` folder** (the folder containing `app.py` and `requirements.txt`). Keep its subfolders and filenames unchanged.

### Step 2. Create a Python environment (recommended)

**Windows Command Prompt:**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If Python is already set up in a compatible environment, you can use that instead.

### Step 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local address shown by Streamlit, normally **http://localhost:8501**.

**Alternative launchers:** On Windows, run `run.bat` after activating the environment and installing dependencies. On macOS/Linux, with the environment activated, run `bash run.sh`. Both launchers start the same Streamlit application.

## 3. Use the application

1. In **About your crop**, check or change the growing environment, root environment, growth stage, and watering method. Initial choices are editable **default assumptions**, not confirmed observations.
2. Upload **one lettuce-leaf photograph** (`.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`; maximum **15 MB**), then tick **“This is a photograph of lettuce.”**
3. Optionally provide the farm location using coordinates, device location, or Pakistan town search; you may also skip location.
4. Optionally enter your observations and any measured values, such as pH, EC, soil moisture, temperatures, or humidity. Leave unknown measurements blank.
5. Select **Get recommendations** to view the image prediction, confidence information, proposed actions, environmental summary, and supporting sources.
6. Expand **Evidence and details** for rule decisions, source links, data provenance, and agent activity.
7. Download the outputs using **Download readable report** (`lettuce_recommendations.html`), **Download report text** (`lettuce_recommendations.md`), or **Download agent record** (`lettuce_agent_record.json`).

If you modify the form after generating a result, select **Get recommendations** again to refresh it.

## 4. Selected classifier

| Property | Included implementation |
| --- | --- |
| Architecture | EfficientNet-B0, ten classification categories |
| Training configuration | No random erasing + balanced classifier retraining (cRT) |
| Selected training seed | `42` |
| Selected encoder epoch / head epoch | `81` / `10` (study-selected checkpoint) |
| Weights file | `models/noerase_crt_balanced/seed_42/best_weights.pt` |
| Model ID | `efficientnet_b0_noerase_crt_balanced_seed42` |
| Checkpoint SHA-256 | `0c9a1ec932d57a10ead1ddd02072cb39a1716d91ac332d5594a799dfacc9033e` |
| Input size | 224 × 224 RGB after resizing and centre cropping |
| Inference mode | Single checkpoint; CPU or automatic GPU selection |
| Reported probability status | Raw softmax scores; no fitted calibration |
| Provisional confidence policy | Maximum score ≥ 0.80 and top-two margin ≥ 0.15 |

The source verifies the checkpoint identity and class ordering before inference. The deployment manifest records **partial confirmation of the original training transform**: image size and embedded ImageNet normalization are confirmed, while the exact decoder/resize/crop settings are recorded as inherited assumptions. Thus packaging this checkpoint does not, by itself, establish complete training-to-deployment preprocessing equivalence.

**Classes:** Anthracnose; Bacterial; Bottom Rot; Downy Mildew; Healthy; Lettuce Mosaic Virus; Powdery Mildew; Septoria Blight; Viral; Wilt and Leaf Blight.

## 5. Application components and data sources

The five deterministic agents are **Image Quality**, **Diagnosis**, **Environmental Context**, **Recommendation**, and **Reporting**. Environmental context affects care-action eligibility and ranking; it does not change the classifier's image probabilities.

The bundled knowledge base includes **33 care actions**, **8 plant-protection reference cards**, **22 source records**, and **17 decision rules**. Recommendations are source-linked where applicable, and the system distinguishes general-care advice from specific plant-protection reference options. These references are **not** proof of local product approval or of treatment effectiveness.

The application uses **Open-Meteo** for outdoor weather and provides bundled Pakistan soil background layers, with online **SoilGrids** lookup where supported. Electrical conductivity background data require a suitable local layer or a supplied measurement. These geographic estimates are not substitutes for measurements taken in a particular field, greenhouse, substrate, or hydroponic solution. Missing or unavailable values remain identifiable in the report.

## 6. Project structure

```text
pakistan_lettuce_dss/
├── app.py                          # Streamlit interface
├── run.bat                         # Windows launcher
├── run.sh                          # macOS / Linux launcher
├── requirements.txt                # Python dependencies
├── .streamlit/config.toml          # Streamlit appearance and settings
├── config/                         # Model, application, policy, soil settings
├── lettuce_dss/                    # Classifier, agents, rules, providers, reporting
├── knowledge/treatment_knowledge_base.json
├── data/soilmaps/                  # Packaged Pakistan soil-map subsets
└── models/noerase_crt_balanced/seed_42/best_weights.pt
```

The `runtime/` subdirectories used for temporary files or caches may be created when the application runs.

## 7. What this release does and does not reproduce

**Included:** selected application model checkpoint, model-identity and integrity checks, inference and preprocessing implementation, five-agent workflow, knowledge base, local soil maps, Streamlit interface, and report exports.

**Not included:** original source-image datasets, processed dataset and partition manifests, notebooks/scripts for all five training configurations, all seed checkpoints, saved per-image test predictions, or the complete evidence needed to independently regenerate the manuscript's experiment tables and scenario counts. For the eight original public data sources, see the dataset references in the accompanying manuscript.

Do not interpret a successful application launch as a reproduction of the paper's reported classification metrics or recommendation-validation results.

## 8. Troubleshooting

| Issue | What to check |
| --- | --- |
| `No module named streamlit` | Activate the intended environment and rerun `python -m pip install -r requirements.txt`. |
| `torch` / `torchvision` import error | Install compatible PyTorch and torchvision builds in the same environment; use the official PyTorch installation selector if a platform-specific build is needed. |
| Checkpoint missing or integrity-check failure | Keep `models/noerase_crt_balanced/seed_42/best_weights.pt` unchanged and restore it from the original ZIP. |
| **Get recommendations** is disabled | Upload an image, confirm that it is lettuce, and check that the model and knowledge base load. |
| Weather or town search is unavailable | Check internet connectivity, or enter coordinates manually; unavailable services should not be treated as measured data. |
| No soil estimate or treatment reference appears | Geographic coverage or eligibility criteria may not be met. Review the report's missing-context and source explanations. |
| No GPU is available | Under **Model settings**, select CPU (or leave **Automatic** to fall back to CPU). |

## 9. Research scope and limitations

This software supports an **initial image-based assessment and contextual care guidance**, not a pathogen-confirmed diagnosis or an independently validated treatment prescription. Confidence scores are not calibrated risk estimates; unseen diseases and non-lettuce inputs are not specifically validated. Treatment and product decisions require appropriate local agricultural expertise and authorization.

The manuscript reports classification and constructed-rule evaluations separately. Agricultural suitability, farmer usability, field outcomes, and complete upload-to-report latency have not been established by this release.

## 10. Citation

When referencing the research, cite the accompanying manuscript:

> *Two-Stage EfficientNet-B0 Classification with Contextual Care Guidance for Lettuce.*

For dataset origins, use the **eight original public dataset references cited in Section 3.1** of that manuscript. The original datasets are not redistributed in this release.

For code availability, link directly to the repository or tagged release containing **this v1.3.2 package and its seed-42 balanced-cRT checkpoint**. Do not link to an older repository version containing a different model as if it reproduced the manuscript.
