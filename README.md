# app-bem-v2

Brainlife app to compute a BEM (Boundary Element Model) from a FreeSurfer recon-all output.

## Inputs

| Key | Datatype | Description |
|-----|----------|-------------|
| `freesurfer` | `neuro/freesurfer` | FreeSurfer subject directory (from recon-all) |

## Outputs

| Key | Datatype | Description |
|-----|----------|-------------|
| `out_dir/bem-sol.fif` | `neuro/bem` | BEM conductor model for forward modelling |

## Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `n_layers` | `3` | Number of BEM layers: `3` for EEG or MEG+EEG (brain/skull/skin), `1` for MEG-only (inner skull only) |
| `ico` | `4` | Icosahedron subdivision order for BEM surfaces (4 = 2562 vertices, 5 = 10242 vertices) |

## Pipeline position

```
app-freesurfer → [app-bem-v2] → app-forward-v2
```

## Container

`docker://brainlife/mne-freesurfer:7.3.2-1.2.1`
