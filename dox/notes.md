# Futuristica — Notes & Experiments

## Results so far

| Run      | Final loss | Stats                                |
|----------|------------|--------------------------------------|
| Siren    | 0.000674   | claude siren, 8x16, ~24h             |
| E16      | 0.000383   | new encoding, long run               |
| Encoding | 0.003700   | encoding:3, 60m, ckp                 |
| Raw      | 0.006588   | encoding:0, 10m                      |

Live on ShaderToy:
- [w3l3W4](https://www.shadertoy.com/view/w3l3W4) — loss < 0.004, ~1h
- [sflXWr](https://www.shadertoy.com/view/sflXWr) — SIREN comparison, loss ~0.000674
- [WXjGzW](https://www.shadertoy.com/view/WXjGzW) — E16 version

## Experiments

Things tried `(/)`, not tried `( )`, failed or abandoned `(x)`:

1.  better image pre-processing (x)
2.  varying the test inputs (/)
3.  changing around the loss function (/)
4.  tweaks to the training method (/)
5.  try it on different images (/)
6.  reload weights from a prior run (/)
7.  train the encoder instead of having a static method ( )
8.  modify the model architecture — attention? ( )
9.  try smaller models (/)
10. conditioning (?)
11. more optimized generated code (/)
12. minibatch training (/) — big win, much faster convergence
13. plateau detection (/) — `--plateau N` stops after N epochs without improvement
14. cosine annealing LR (/) — single cycle over full run, much better than short cycles

## Architecture notes

The positional encoding (`lib/trainer.py: positional_encoding`) maps (x,y) → 16-wide
feature vector that feeds the MLP. All three `--mapping` modes produce the same width
so the network input size is constant. `lib/translate.py` mirrors this in GLSL exactly.

`polar` (default): hint slots = [sin θ, cos θ] — no atan2 discontinuity, recommended.
`fourier`: pure spectral, no spatial hint. Classic NeRF encoding.
`legacy`: abs(fract()) kink-based — kept for reproducibility, not recommended.

Sweet spot: 16-wide × 4 layers. 6-8 layers gets sharper but may be too heavy for
mobile WebGL. `--four` adds a 4th output channel (grayscale ratio) for richer colour.

## SSH / headless

Training on a remote machine? Always use `--no_gui` (run-training and run-grid set this
automatically). The matplotlib window and `bin/trainspotting` both need a display.
To view progress from SSH, use `bin/historic-output` which doesn't need X11,
or `scp`/`rsync` the images dir locally.

## Dependencies

Core: Python 3.10+, PyTorch, Pillow, NumPy, matplotlib, torchvision (for perceptual loss).
See `requirements.txt`.

Helper scripts need: `ffmpeg` (mp4 creation), `mplayer` (mp4 playback), `xv` (png preview).
All three are optional — training works fine without them.

## Further reading

- https://medium.com/deepfail/implementing-jpeg-compression-in-pytorch-b0a830889f59
- https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- https://www.geeksforgeeks.org/pytorch-loss-functions/
- Sitzmann et al. 2020 — "Implicit Neural Representations with Periodic Activation Functions" (SIREN paper)
