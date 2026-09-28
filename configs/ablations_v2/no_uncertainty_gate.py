_base_ = ['../experiments/rpgv_v2_joint.py']
# U=1, retaining the same bounded correction and SDF supervision.
model = dict(use_uncertainty_gate=False)
