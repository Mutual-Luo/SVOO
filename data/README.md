# Data Layout

`example/` is for quick inference demos.

```text
data/example/<id>/prompt.txt
data/example/<id>/image.jpg
```

`profile_data/` is for offline sparsity profile generation.

```text
data/profile_data/prompt.txt      # one prompt per line
data/profile_data/image/1.jpg     # image for line 1, used by i2v profiles
data/profile_data/image/2.jpg     # optional image for line 2, if present
```

All non-empty prompt lines are profiled. For image-to-video profiling, line `N`
uses `data/profile_data/image/N.jpg`.

The loader also accepts `.jpeg`, `.png`, or `image/N/image.{jpg,jpeg,png}`.

The demo examples and profiling prompts are intentionally separate, even when
they happen to contain the same content.
