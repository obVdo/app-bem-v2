"""
app-bem-v2: Compute BEM (Boundary Element Model) from FreeSurfer output.

Inputs : FreeSurfer subject directory (from recon-all).
Outputs: bem-sol.fif (BEM conductor model for forward modelling).
"""

import os
import sys

# Set up FreeSurfer environment (same as sourcing SetUpFreeSurfer.sh)
if not os.environ.get('FREESURFER_HOME'):
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
fs_home = os.environ['FREESURFER_HOME']
os.environ['PATH'] = os.path.join(fs_home, 'bin') + ':' + os.environ.get('PATH', '')

# Resolve brainlife_utils — try local copy first, then parent monorepo
app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt

import mne

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == RESOLVE FREESURFER DIRECTORY ==
fs_path      = config.get('freesurfer') or config.get('output')
subjects_dir = config.get('subjects_dir')
subject      = config.get('subject')

if fs_path and os.path.isdir(fs_path):
    fs_path = os.path.abspath(fs_path)
    if not subjects_dir:
        subjects_dir = os.path.dirname(fs_path)
    if not subject:
        subject = os.path.basename(fs_path)

if not subjects_dir or not subject:
    add_info_to_product(
        report_items,
        "FATAL: No FreeSurfer directory found. "
        "Set 'freesurfer' in config.json to the subject's FreeSurfer directory.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

if not os.path.isdir(os.path.join(subjects_dir, subject)):
    add_info_to_product(
        report_items,
        f"FATAL: FreeSurfer subject directory not found: "
        f"{os.path.join(subjects_dir, subject)}",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

add_info_to_product(report_items, f"Subject: {subject} | subjects_dir: {subjects_dir}", "info")

# == PARAMETERS ==
n_layers_raw = config.get('n_layers') or '3'
n_layers = int(n_layers_raw)

ico_raw = config.get('ico')
ico = int(ico_raw) if ico_raw not in (None, '', 'None', 'none') else None

if n_layers == 3:
    conductivity = (0.3, 0.006, 0.3)
elif n_layers == 1:
    conductivity = (0.3,)
else:
    add_info_to_product(
        report_items,
        f"FATAL: n_layers must be 1 or 3 (got {n_layers}). "
        "Use 3 for EEG or combined MEG+EEG, 1 for MEG-only.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

add_info_to_product(
    report_items,
    f"BEM parameters: {n_layers}-layer | ico={ico} | conductivity={conductivity} S/m",
    "info"
)

# == WATERSHED BEM SURFACES ==
# make_watershed_bem creates inner_skull.surf, outer_skull.surf, outer_skin.surf
# in {subjects_dir}/{subject}/bem/ (as copies).
# Skip watershed if surfaces already exist.
bem_dir    = os.path.join(subjects_dir, subject, 'bem')
inner_surf = os.path.join(bem_dir, 'inner_skull.surf')
outer_skull_surf = os.path.join(bem_dir, 'outer_skull.surf')
outer_skin_surf  = os.path.join(bem_dir, 'outer_skin.surf')

if n_layers == 3:
    surfaces_exist = (
        os.path.isfile(inner_surf)
        and os.path.isfile(outer_skull_surf)
        and os.path.isfile(outer_skin_surf)
    )
else:
    surfaces_exist = os.path.isfile(inner_surf)

if surfaces_exist:
    add_info_to_product(report_items, "BEM surfaces found — skipping watershed step.", "info")
else:
    add_info_to_product(report_items, "Running FreeSurfer watershed BEM...", "info")
    try:
        mne.bem.make_watershed_bem(subject, subjects_dir, overwrite=True, verbose=True)
        add_info_to_product(report_items, "Watershed BEM surfaces created.", "info")
    except Exception as e:
        add_info_to_product(
            report_items,
            f"FATAL: Watershed BEM failed: {e}\n"
            "Ensure FreeSurfer is installed and recon-all has completed.",
            "error"
        )
        create_product_json(report_items)
        sys.exit(1)

# == MAKE BEM MODEL ==
try:
    model = mne.make_bem_model(
        subject, ico=ico, conductivity=conductivity,
        subjects_dir=subjects_dir, verbose=True
    )
    n_triangles = sum(s['ntri'] for s in model)
    add_info_to_product(
        report_items,
        f"BEM model: {len(model)} surface(s), {n_triangles} triangles total",
        "info"
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: make_bem_model failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == MAKE BEM SOLUTION ==
try:
    bem_sol = mne.make_bem_solution(model, verbose=True)
    add_info_to_product(report_items, "BEM solution computed.", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: make_bem_solution failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == SAVE BEM SOLUTION ==
bem_path = os.path.join('out_dir', 'bem-sol.fif')
try:
    mne.write_bem_solution(bem_path, bem_sol, overwrite=True)
    add_info_to_product(report_items, f"Saved: {bem_path}", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not save bem-sol.fif: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == QC FIGURES — BEM surfaces on MRI slices ==
# brain_surfaces="white" overlays white matter boundary for anatomical context.
# Falls back to no brain surface overlay if the white surface is missing.
_brain_surfaces = "white"
if not os.path.isfile(os.path.join(subjects_dir, subject, 'surf', 'lh.white')):
    _brain_surfaces = None

fig_paths = {}
for orientation in ('coronal', 'axial', 'sagittal'):
    try:
        fig = mne.viz.plot_bem(
            subject=subject, subjects_dir=subjects_dir,
            brain_surfaces=_brain_surfaces,
            orientation=orientation, show=False
        )
        fig_path = os.path.join('out_figs', f'bem_{orientation}.png')
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        fig_paths[orientation] = fig_path
    except Exception as e:
        add_info_to_product(
            report_items, f"Could not plot BEM ({orientation}): {e}", "warning"
        )

# Only embed coronal in product.json (keep file small); all views go in report.html
if 'coronal' in fig_paths:
    add_image_to_product(report_items, 'BEM surfaces — coronal', filepath=fig_paths['coronal'])

# == SAVE REPORT ==
report = mne.Report(title='BEM Report')
for orientation, fig_path in fig_paths.items():
    report.add_image(fig_path, title=f'BEM surfaces — {orientation}')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "BEM computation completed successfully.", "success")
create_product_json(report_items)
print("Done.")
