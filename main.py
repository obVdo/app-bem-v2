"""
app-bem-v2: Compute BEM (Boundary Element Model) from FreeSurfer output.

Inputs : FreeSurfer subject directory (from recon-all).
Outputs: bem-sol.fif (BEM conductor model for forward modelling).
"""

import os
import sys
import numpy as np

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
ensure_output_dirs('out_dir', 'out_figs', 'out_dir_report')
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

add_info_to_product(report_items, f"Subject: {subject}", "info")

# == PARAMETERS ==
n_layers_raw = config.get('n_layers') or '3'
n_layers = int(n_layers_raw)

ico_raw = config.get('ico')
ico = int(ico_raw) if ico_raw not in (None, '', 'None', 'none') else None

bem_expansion_raw = config.get('bem_expansion_mm')
bem_expansion_mm = float(bem_expansion_raw) if bem_expansion_raw not in (None, '', 'None', 'none') else 8.0

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
    add_info_to_product(report_items, "Running FreeSurfer watershed BEM...", "info")
    try:
        import subprocess
        subprocess.run(
            ['mne', 'watershed_bem', '-s', subject, '-d', subjects_dir, '-o', '-a'],
            check=True
        )
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
else:
    add_info_to_product(report_items, "Running FreeSurfer watershed BEM...", "info")
    try:
        import subprocess
        subprocess.run(
            ['mne', 'watershed_bem', '-s', subject, '-d', subjects_dir, '-o', '-a'],
            check=True
        )
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

_brain_surfaces = "white" if os.path.isfile(
    os.path.join(subjects_dir, subject, 'surf', 'lh.white')) else None

def _expand_anterior(coords, mm):
    """Push outer skull/skin vertices outward, anterior-weighted.
    Vertices facing anteriorly (+y in RAS) are pushed by up to mm mm;
    the effect tapers to zero toward the sides and back.
    This fixes watershed failures where defaced T1 anterior skull
    has poor contrast, causing outer skull/skin to collapse inward.
    """
    centroid = coords.mean(axis=0)
    directions = coords - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_dirs = directions / norms
    weight = np.clip(np.dot(unit_dirs, np.array([0., 1., 0.])), 0, None)
    return coords + unit_dirs * (weight[:, None] * mm)

report = mne.Report(title='BEM Report')

# == MAKE BEM MODEL ==
bem_ok = False
model = None
try:
    model = mne.make_bem_model(
        subject, ico=ico, conductivity=conductivity,
        subjects_dir=subjects_dir, verbose='DEBUG'
    )
    n_triangles = sum(s['ntri'] for s in model)
    add_info_to_product(
        report_items,
        f"BEM model: {len(model)} surface(s), {n_triangles} triangles total",
        "info"
    )
    bem_ok = True
except Exception as e:
    if 'not completely inside' in str(e):
        add_info_to_product(
            report_items,
            f"BEM surface intersection detected — retrying with anterior expansion "
            f"({bem_expansion_mm}mm). Likely caused by poor skull contrast in defaced T1.",
            "warning"
        )
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        outer_skull_path = os.path.join(bem_dir, 'outer_skull.surf')
        outer_skin_path  = os.path.join(bem_dir, 'outer_skin.surf')
        try:
            skull_coords, skull_tris = mne.read_surface(outer_skull_path)
            skin_coords,  skin_tris  = mne.read_surface(outer_skin_path)

            # Capture BEFORE — add to shared report + single slice image
            _before_img = None
            try:
                report.add_bem(subject=subject, subjects_dir=subjects_dir,
                    title='BEM surfaces — BEFORE expansion (intersection)', decim=4, width=512)
            except Exception:
                pass
            try:
                _fig = mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                    brain_surfaces=_brain_surfaces, orientation='axial',
                    slices=[137], show=False)
                _fig.canvas.draw()
                _before_img = np.asarray(_fig.canvas.buffer_rgba())[..., :3]
                plt.close(_fig)
            except Exception:
                pass

            mne.write_surface(outer_skull_path, _expand_anterior(skull_coords, bem_expansion_mm), skull_tris, overwrite=True)
            mne.write_surface(outer_skin_path,  _expand_anterior(skin_coords,  bem_expansion_mm), skin_tris,  overwrite=True)

            model = mne.make_bem_model(
                subject, ico=ico, conductivity=conductivity,
                subjects_dir=subjects_dir, verbose=False
            )
            n_triangles = sum(s['ntri'] for s in model)
            add_info_to_product(
                report_items,
                f"BEM model after expansion: {len(model)} surface(s), {n_triangles} triangles total",
                "info"
            )
            bem_ok = True

            # Capture AFTER plot and save before/after comparison
            try:
                _fig = mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                    brain_surfaces=_brain_surfaces, orientation='axial',
                    slices=[137], show=False)
                _fig.canvas.draw()
                _after_img = np.asarray(_fig.canvas.buffer_rgba())[..., :3]
                plt.close(_fig)

                if _before_img is not None:
                    _comp, _axs = plt.subplots(1, 2, figsize=(10, 5))
                    _axs[0].imshow(_before_img); _axs[0].set_title('Before expansion\n(intersection)', fontsize=10, color='red'); _axs[0].axis('off')
                    _axs[1].imshow(_after_img);  _axs[1].set_title(f'After +{bem_expansion_mm}mm anterior\n(fixed)', fontsize=10, color='lime'); _axs[1].axis('off')
                    _comp.patch.set_facecolor('black')
                    _comp_path = os.path.join('out_figs', 'bem_expansion_fix.png')
                    _comp.savefig(_comp_path, dpi=100, bbox_inches='tight', facecolor='black')
                    plt.close(_comp)
                    add_image_to_product(report_items, 'BEM expansion fix — axial slice 137', filepath=_comp_path)
            except Exception:
                pass

        except Exception as e2:
            add_info_to_product(
                report_items,
                f"WARNING: make_bem_model failed even after expansion: {e2} — check surfaces in report.",
                "error"
            )
    else:
        add_info_to_product(
            report_items,
            f"WARNING: make_bem_model failed: {e} — check surfaces in report.",
            "error"
        )

# == MAKE BEM SOLUTION (only if model succeeded) ==
if bem_ok:
    try:
        bem_sol = mne.make_bem_solution(model, verbose=True)
        add_info_to_product(report_items, "BEM solution computed.", "info")
    except Exception as e:
        add_info_to_product(report_items, f"FATAL: make_bem_solution failed: {e}", "error")
        bem_ok = False

# == SAVE BEM SOLUTION ==
if bem_ok:
    bem_path = os.path.join('out_dir', 'meg.fif')  # meg/fif datatype requires meg.fif
    try:
        mne.write_bem_solution(bem_path, bem_sol, overwrite=True)
        add_info_to_product(report_items, f"Saved: {bem_path}", "info")
    except Exception as e:
        add_info_to_product(report_items, f"FATAL: Could not save BEM solution: {e}", "error")
        bem_ok = False

# == QC FIGURE FOR PRODUCT.JSON — middle slice from all 3 orientations, side-by-side ==

def _crop_middle_slice(fig):
    """Render fig, crop out the middle subplot, return as RGB array."""
    axes = fig.axes
    mid_ax = axes[len(axes) // 2]
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    full_img = np.asarray(buf)[..., :3]
    bbox = mid_ax.get_position()
    h, w = full_img.shape[:2]
    x0, x1 = int(bbox.x0 * w), int(bbox.x1 * w)
    y0, y1 = int((1 - bbox.y1) * h), int((1 - bbox.y0) * h)
    return full_img[y0:y1, x0:x1]

slices = {}
for orientation in ('coronal', 'axial', 'sagittal'):
    try:
        fig = mne.viz.plot_bem(
            subject=subject, subjects_dir=subjects_dir,
            brain_surfaces=_brain_surfaces,
            orientation=orientation, show=False
        )
        slices[orientation] = _crop_middle_slice(fig)
        plt.close(fig)
    except Exception as e:
        add_info_to_product(report_items, f"Could not plot BEM ({orientation}): {e}", "warning")

if slices:
    n = len(slices)
    combined, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (orientation, img) in zip(axes, slices.items()):
        ax.imshow(img)
        ax.set_title(orientation, fontsize=10)
        ax.axis('off')
    thumb_path = os.path.join('out_figs', 'bem_thumb.png')
    combined.savefig(thumb_path, dpi=100, bbox_inches='tight')
    plt.close(combined)
    add_image_to_product(report_items, 'BEM surfaces — middle slice', filepath=thumb_path)

# == SAVE REPORT — interactive slider via report.add_bem() ==
# add_bem generates a slider through all MRI slices with BEM contours.
# decim=4 → ~60 slices; width=512 is standard MRI resolution.
try:
    report.add_bem(
        subject=subject, subjects_dir=subjects_dir,
        title='BEM surfaces — AFTER expansion (fixed)' if os.path.isfile(os.path.join('out_figs', 'bem_expansion_fix.png')) else 'BEM surfaces (interactive)',
        decim=4, width=512
    )
except Exception as e:
    add_info_to_product(report_items, f"Could not add interactive BEM to report: {e}", "warning")
    if os.path.isfile(os.path.join('out_figs', 'bem_thumb.png')):
        report.add_image(os.path.join('out_figs', 'bem_thumb.png'), title='BEM surfaces')

_expansion_fig = os.path.join('out_figs', 'bem_expansion_fix.png')
if os.path.isfile(_expansion_fig):
    report.add_image(_expansion_fig, title='BEM expansion fix — before/after (axial 137)')
report.save(os.path.join('out_dir_report', 'report.html'), overwrite=True)

if bem_ok:
    add_info_to_product(report_items, "BEM computation completed successfully.", "success")
else:
    add_info_to_product(report_items, "BEM surfaces generated but model failed — inspect report images.", "warning")
create_product_json(report_items)
print("Done.")
