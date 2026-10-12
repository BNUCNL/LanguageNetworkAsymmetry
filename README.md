# LanguageNetworkAsymmetry

Code for **Developmental dissociation of hemispheric asymmetries in functional connectivity organization and activation in the language network**, submitted to *NeuroImage*.

The study includes 1167 right-handed children and adolescents from the Chinese Child Brain Development (CCBD) cohort. Scripts are ordered from data preparation to the main analyses, supplementary analyses, and figures. Data preparation starts from supplied aligned gradients and first-level task beta outputs.

| Script | Content |
| --- | --- |
| `01_data_preparation.py` | Sample selection, reading data, retained task runs, and head motion; optional beta extraction |
| `02_asymmetry_extraction.py` | FC organization and activation asymmetries, using L14−R9 and nine homologous parcel pairs |
| `03_developmental_dissociation.py` | Age patterns and connectivity–activation associations (Results 3.1) |
| `04_reading_associations.py` | Reading age patterns and the two brain–reading associations (Results 3.2) |
| `05_moderated_mediation.py` | Age-moderated indirect associations with reading (Results 3.3) |
| `06_supplementary_analysis.py` | Homologous-parcel analysis and first-ten diffusion eigenvalue contributions |
| `07_plot_brain_materials.py` | Fig1 brain maps, FC matrix, and atlas illustrations |
| `08_plot_results.py` | Fig2 A–D, Fig3 B–D, Fig4 B, and Supplementary Figs1–2 |

`stats.py` contains the shared statistical functions. Fig3A, Fig4A, and the final figure layouts were prepared in PowerPoint.

## Run

```bash
python -m pip install -r requirements.txt
cp config.example.json config.local.json
cp brain_materials.example.json brain_materials.local.json
cp figures.example.json figures.local.json
```

Set the file paths in the local configuration files, then run:

```bash
python 01_data_preparation.py --config config.local.json --output-dir ../study_results/prepared
python 02_asymmetry_extraction.py --config config.local.json --output-dir ../study_results/metrics
python 03_developmental_dissociation.py --config config.local.json --output-dir ../study_results/asymmetry
python 04_reading_associations.py --config config.local.json --output-dir ../study_results/reading
python 05_moderated_mediation.py --config config.local.json --output-dir ../study_results/mediation --bootstraps 10000 --seed 42
python 06_supplementary_analysis.py --config config.local.json --output-dir ../study_results/supplementary_9v9 --main-results-dir ../study_results/asymmetry
python 06_supplementary_analysis.py --config config.local.json --output-dir ../study_results/spectrum --spectrum-only
python 07_plot_brain_materials.py --config brain_materials.local.json --output-dir ../study_results/fig1
python 08_plot_results.py --config figures.local.json --output-dir ../study_results/figures
```

To extract beta values from first-level task outputs before preparing the sample:

```bash
python 01_data_preparation.py extract-beta --config config.local.json --output-dir ../study_results/task_betas
```

Relative paths resolve from the configuration file. Use new output directories. Figure scripts produce separate 600 dpi TIFF panels; result-panel text is 12 pt.
