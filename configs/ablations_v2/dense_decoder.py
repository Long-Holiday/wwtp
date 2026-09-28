_base_ = ['../experiments/rpgv_v2_joint.py']
# Same 64 channels and contour head; only replace the decoder implementation.
# Train independently with the single-stage joint protocol.
model = dict(decoder_type='dense')
