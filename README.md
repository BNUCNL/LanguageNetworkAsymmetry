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

Run from the repository root. The scripts contain the relative paths to the project data and atlas files. Analysis outputs go to `results/`, and figure materials go to `figures/`.

```bash
python -m pip install -r requirements.txt
python 01_data_preparation.py
python 02_asymmetry_extraction.py
python 03_developmental_dissociation.py
python 04_reading_associations.py
python 05_moderated_mediation.py
python 06_supplementary_analysis.py
python 06_supplementary_analysis.py --spectrum-only
python 07_plot_brain_materials.py
python 08_plot_results.py
```

Use new output directories; existing results are not overwritten. Figure scripts produce separate 600 dpi TIFF panels with 12 pt result-panel text.

