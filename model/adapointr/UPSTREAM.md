# AdaPoinTr upstream code

Source: https://github.com/yuxumin/PoinTr

Pinned commit: `4603257ed3db9e7dad349b712e1b2fe0da207015`.

`AdaPoinTr.py` and `Transformer_utils.py` are copied from `models/` at that
commit. The upstream MIT license is included as `LICENSE`. The source files
retain their original layout to make upstream comparisons straightforward.

Local changes to `AdaPoinTr.py` are limited to:

- Local dependency imports and removal of the unrelated model-registry decorator.
- Optional `input_feature_dim` and `features` arguments: bypass the grouper and
  project the provided keypoint features instead of extracting them from raw XYZ.
- Optional `return_features`: expose the normal query features alongside the
  original output tuple, so CleanPose can use them for NOCS and pose estimation.
- Removal of the equal point-count assertion in the native `get_loss` helper.
  The active network uses its own completion loss with 1,024 output points.

Only dependency imports change in `Transformer_utils.py`. `compat.py` maps
PointNet2 operations to CleanPose's existing CUDA extension, provides CPU test
fallbacks, and implements the same symmetric mean Euclidean Chamfer distance
using PyTorch instead of building PoinTr's separate Chamfer extension. Its FPS
and jitter helpers follow upstream `utils/misc.py`.

The encoder, decoder, graph/deformable attention implementations, query ranking,
FC/folding heads, 64-query denoising branch and `get_loss` computations are
retained. In particular, upstream's ranking uses discrete `argsort`; it has no
gradient to ranking parameters, and no additional score gate is introduced.

The active network has one path. `model/compose_completion.py` reuses only the
vendored encoder and attention operators and implements ComPose-style scored
selection, keypoint refinement and local expansion, plus sampled coordinate-axis
proxies. Its custom decoder jointly processes 64 surface queries and 48 axis
queries in global attention; local graph attention only processes the surface
queries. Encoder memory consists only of the 96 detected keypoints. Axis heads
generate point coordinates directly, followed by axis-to-pose readout and refinement.
The old native adapter and mode selection have been removed. The upstream full
model remains in this vendor file for provenance; it is not used by Net.
See `docs/completion.md` for the current network and its real-data validation.
