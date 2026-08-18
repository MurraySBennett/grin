# Structured methodological survey of empirical GRT studies

## Purpose

This survey asks a software-scope question rather than attempting a general history
of General Recognition Theory (GRT):

> How often, and under what data-collection regimes, has empirical GRT research used
> participant-level 2 x 2 factorial identification data that GRIN can analyse?

Secondary aims are to (a) compare published trial-count and accuracy regimes with
GRIN's validated training envelope, (b) document common analysis software and model
selection practices, and (c) locate shareable participant-level confusion matrices
for an empirical case study.

The intended report is a **structured methodological survey**, not a formal systematic
review or meta-analysis. Any manuscript description should use that label unless the
search is subsequently registered, independently screened, and completed under a
full systematic-review protocol.

## Eligibility criteria

### Include in the methodological survey

1. An empirical human study that explicitly applies GRT, multidimensional signal
   detection theory, GRT-wIND, or a directly derived GRT test.
2. The task produces identification, classification, concurrent-rating, or Garner
   data relevant to dimensional interaction.
3. The report contains enough information to identify the task design and at least
   one analysis target (PI, PS, DS, dimensional interaction, or a fitted GRT space).
4. Journal articles, dissertations, chapters with original data, and accessible
   preprints are eligible; publication type is coded.

### Direct-GRIN subset

A study is directly compatible only when it reports a 2 x 2 full-factorial
identification task with four stimulus classes and four joint responses, analysed at
the participant level or with recoverable participant-level counts. Decisional
separability need not have been assumed by the original authors, but GRIN's own DS
assumption must be noted when reanalysing it.

### Exclude from the empirical inventory

- Purely theoretical, simulation-only, tutorial, or software papers with no original
  empirical application (retain separately as background sources).
- Uses of "GRT" meaning an unrelated theory.
- Categorisation/decision-bound studies that cite GRT generically but do not analyse
  dimensional interaction using a GRT method.
- Nonhuman studies, unless later judged relevant to software scope.

## Search strategy

Search date for the initial pass: 2026-08-13.

Databases recommended for the definitive pass:

- PsycINFO
- Web of Science or Scopus
- Google Scholar for citation chasing and grey literature
- Crossref/OpenAlex for metadata and deduplication
- ProQuest Dissertations & Theses if dissertation coverage is desired

Core keyword searches:

1. `"general recognition theory" AND (experiment* OR empirical OR identification)`
2. `"general recognition theory" AND ("confusion matrix" OR separability OR independence)`
3. `("multidimensional signal detection" OR mdsdt) AND identification AND perception`
4. `("GRT-wIND" OR "GRTwIND") AND (application OR experiment*)`
5. `(grtools OR mdsdt) AND (study OR experiment OR data)`

Run backward-reference and forward-citation searches from:

- Ashby and Townsend (1986)
- Thomas (2001)
- Silbert and Thomas (2013)
- Soto et al. (2015; GRT-wIND)
- Silbert and Hawkins (2016; mdsdt tutorial)
- Soto et al. (2017; grtools)

The exact query, database, date, result count, and export filename should be recorded
in a search log before screening begins. Database exports should be retained in the
archived research materials.

## Screening procedure

1. Deduplicate by DOI, then title/author/year.
2. Title/abstract screen for an empirical application of GRT.
3. Full-text screen for task design and analysis method.
4. Assign one disposition:
   - `include_direct`: directly compatible with GRIN;
   - `include_adjacent`: empirical GRT but not directly compatible;
   - `background`: theoretical/tutorial/software source;
   - `exclude`: not an empirical GRT application;
   - `unclear`: requires full text or supplementary material.
5. A second reviewer should independently check all `include_direct` records and a
   sample of exclusions before manuscript claims are based on the inventory.

## Coding variables

The working CSV lives at `data/literature/grt_study_screening.csv`. For each study,
code:

- bibliographic identity and DOI/URL;
- substantive domain and stimulus dimensions;
- task paradigm and factorial size;
- sample size and unit of analysis;
- trials per participant and per stimulus;
- balanced/unbalanced presentation;
- reported accuracy range;
- analysis method and software;
- whether PI, PS, and DS were addressed;
- whether uncertainty, convergence failures, or exclusions were reported;
- availability of trial data or participant confusion matrices;
- direct GRIN compatibility and, if not, why not;
- page/table/supplement location supporting every extracted number.

Use `NR` for checked-but-not-reported, `NA` for not applicable, and leave a field
blank only when it has not yet been checked. Do not infer trial counts from vague
descriptions without recording the calculation in `notes`.

## Planned outputs

1. A flow count of records found, deduplicated, screened, and included.
2. A supplementary table containing every included empirical application.
3. A compact main-text summary of task designs, trial-count regimes, methods, and
   public-data availability.
4. A direct comparison between published regimes and GRIN's validated envelope.
5. A shortlist of participant-level datasets for empirical reanalysis.

## Initial observations (not final survey results)

- The 2 x 2 task is described by the grtools tutorial as the field's most widely
  used design, but that statement should be tested against the completed inventory.
- Soto et al. (2015) reports 600 trials per participant before removal of a
  participant-specific learning phase; individual confusion matrices are stated to
  be in the supplementary material. This is the strongest initial reanalysis lead.
- Published empirical work also includes 3 x 3 identification, GRT-wIND group fits,
  multilevel speech models, concurrent ratings, and Garner designs. These establish
  the wider GRT landscape but are not automatically compatible with GRIN.
- Trial counts, participant exclusions, and accuracy distributions are often buried
  in methods or supplements, so abstract-only screening cannot answer the envelope
  question.

