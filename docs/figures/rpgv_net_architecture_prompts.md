# RPGV-Net architecture image prompts

Generated with the built-in image generation model. The final PNG is `rpgv_net_architecture.png`.

## Base generation

> Create a clean, opaque-white, wide landscape scientific architecture figure suitable for a top-tier computer vision / remote sensing journal. Flat 2D vector-like illustration, crisp charcoal text, thin arrows, no transparency, no dark background, no 3D cubes, no shadows. Title: "RPGV-Net Architecture". Draw three horizontal lanes. RGB lane: Local RGB → Shared MiT-B2 → R1 1/4, R2 1/8, R3 1/16, R4 1/32; global thumbnail uses the same shared MiT-B2, then global token + FiLM modulates all four features; R1 produces RGB boundary and uncertainty cues. Geometry lane: frozen offline Depth Anything V2 → pseudo-depth D0 and reliability Q0 → RGR rectification → corrected depth and task reliability Qd → geometry encoder + top-down FPN → G1 1/4, G2 1/8, G3 1/16, G4 1/32. Exactly two fusion points: R1 and G1 pass through Haar boundary validation and Qd-weighted fusion at 1/4; R3 and G3 pass through deep region validation and Qd-weighted fusion at 1/16. R2 and R4 bypass fusion. Four resulting RGB features → multi-scale decoder → coarse, boundary, SDF heads → boundary refinement → RGB detail refinement → final segmentation mask. Use pale blue for RGB, pale teal for geometry, pale orange for validation, pale violet for fusion. Large legible English labels, generous white space, no invented branches, training losses, or extra fusion points.

## Final edit

> Edit the provided RPGV-Net architecture image while keeping its clean white journal-figure layout, typography, blue/teal/orange/purple palette, and all correct components. Replace the local RGB street image with an overhead wastewater treatment plant with round clarifier tanks; make the global thumbnail show the same site at wider extent. Add guidance from RGB cues to RGR labelled R1 + boundary. Remove direct arrows from G2 or G4 to the decoder; geometry reaches the decoder only through the two validation and Qd-weighted fusion paths. Mark the two MiT-B2 depictions as shared weights. Replace the city output icon with a wastewater treatment plant segmentation mask. Keep all existing labels and correct arrows; do not add modules or losses.

## Topology correction edits

> Correct the geometry-to-validation wiring: G1 enters Haar boundary validation; G3 enters deep region validation. Both are produced by the geometry encoder and FPN. Preserve the Qd gating arrows and RGB feature inputs. Remove an erroneous extra G1-to-region arrow. Preserve all other content.

> Connect the R3 feature to the deep region validator. Connect the 1/4 fusion output to the decoder, labelled fused R1. Show an explicit R2 RGB bypass into the decoder alongside the existing R4 bypass. Remove the direct FiLM-to-decoder connector. Preserve all other modules and labels.
