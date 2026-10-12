# LanguageNetworkAsymmetry

Analysis code for **Developmental dissociation of hemispheric asymmetries in functional connectivity organization and activation in the language network**, submitted to *NeuroImage*.

The scripts follow the Results section. They use participant-level metrics from the 1167 right-handed children and adolescents in the Chinese Child Brain Development (CCBD) cohort.

| Script | Manuscript section | Analyses |
| --- | --- | --- |
| `01_developmental_dissociation.py` | 3.1 | Leftward asymmetry, age patterns, connectivity–activation association and its age moderation; the reported homologous-parcel and gradient-spectrum checks |
| `02_reading_associations.py` | 3.2 | Reading age pattern and the two separate brain–reading associations |
| `03_moderated_mediation.py` | 3.3 | Age-moderated mediation, overall indirect association and the higher-minus-lower-age contrast |

`stats.py` contains the shared transformations and inference functions. The repository covers the statistical analyses; inputs are the aligned-gradient, reading-task beta and reading-assessment metrics prepared as described in Methods 2.1–2.6.

## Run

The analyses were checked with Python 3.14.5 and the versions in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
cp config.example.json config.local.json
```

Set the input paths in `config.local.json`, then run:

```bash
python 01_developmental_dissociation.py --config config.local.json --output-dir ../results/asymmetry
python 02_reading_associations.py --config config.local.json --output-dir ../results/reading
python 03_moderated_mediation.py --config config.local.json --output-dir ../results/mediation --bootstraps 10000 --seed 42
```

Paths resolve relative to the configuration file. Outputs must be outside the code directory, and existing nonempty output directories are not overwritten. The first script also writes `supplementary_9v9` and `spectrum` next to the main asymmetry output when their optional inputs are supplied. Remove the corresponding configuration entries to run only the main models.

## Inputs

`subject_metrics.csv` contains one row per participant, in the analysis order:

| Column | Meaning |
| --- | --- |
| `sub_id` | Participant identifier |
| `Age_continuous` | Age in years |
| `sex_num` | Female = 0, male = 1 |
| `site_code` | Integer acquisition-site code |
| `gradient_mean_asymmetry` | Mean aligned principal gradient in the 14 left language parcels minus the mean in the nine right parcels |
| `rs_minus_cb` | Corresponding left-minus-right mean beta contrast for sentence reading versus checkerboard stimulation |
| `三分钟快速阅读_总正确率（%）` | Overall accuracy on the three-minute silent reading assessment |
| `rest_motion`, `task_motion` | Mean framewise displacement for the two fMRI modalities |

The optional `homotopic_subject_metrics.csv` has the same participants, order and covariates, with the two asymmetry columns calculated from the nine bilateral homologous language-network pairs. It must retain `array_row`, `task_run_n` and `task_motion_run_n` from the main sample. Its directory also contains `homotopic_language_pairs.csv`, with `base_label`, `left_index_1based` and `right_index_1based` columns.

The optional `diffusion_eigenvalues.csv` contains `sub_id` and `lambda_1` through `lambda_10`, before gradient alignment. Component percentages use each eigenvalue divided by the sum of these ten eigenvalues. Fitted-curve correlations are descriptive and have no inferential p values or confidence intervals.

## Statistics

Age is represented by standardized linear age and a centered, orthogonal quadratic term. Continuous measures use sample-SD standardization. Head motion is residualized on both age terms. Sex and acquisition site are included throughout; reading age models have no motion covariate, each brain–reading model uses its corresponding fMRI motion, and connectivity–activation and mediation models use both.

Nonmediation models use ordinary OLS Student-t inference and pointwise 95% confidence intervals. Leftward mean asymmetry is assessed with positive one-sided one-sample t tests. Mediation uses 10000 participant bootstrap resamples stratified by site, seed 42, with identical draws for all paths and contrasts and percentile 95% confidence intervals. Pairwise associations and all mediation paths retain linear and quadratic age moderation. The two brain–reading models are fitted separately. The overall indirect association averages the age-specific products across observed ages; the higher-minus-lower contrast compares the sample mean age plus and minus one SD.

Bonferroni correction covers the two age terms in each age model, five terms in each pairwise model, and 5/8/5 terms in the mediator/outcome/total mediation models. The overall indirect association and higher-minus-lower-age contrast are reported separately without correction. This reporting scheme was finalized during manuscript revision after inspection of the results; it was not preregistered.

## Data access

CCBD participant data are subject to consortium governance and are not included. De-identified summary data supporting the manuscript may be requested from the corresponding author with CCBD consortium approval. External atlas resources must be obtained from their original providers.

Corresponding author: Zonglei Zhen, Beijing Normal University, zhenzonglei@bnu.edu.cn.
