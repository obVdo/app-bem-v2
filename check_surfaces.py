import sys, json, os
import mne

config = json.load(open('config.json'))
fs_path = config.get('freesurfer') or config.get('output', '')
fs_path = os.path.abspath(fs_path)
subject = os.path.basename(fs_path)
subjects_dir = os.path.dirname(fs_path)

result = []
try:
    model = mne.make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                               subjects_dir=subjects_dir, verbose=False)
    msg = f"SUCCESS: surfaces do not intersect. {len(model)} surfaces, {sum(s['ntri'] for s in model)} triangles."
    result.append({"type": "success", "msg": msg})
except Exception as e:
    result.append({"type": "error", "msg": f"FAILED: {e}"})

print(result[0]['msg'])
with open('product.json', 'w') as f:
    json.dump({"brainlife": result}, f)
