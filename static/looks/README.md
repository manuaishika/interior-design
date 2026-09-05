# Reference photographs

Drop real photographs in here and they appear in the Studio's look popup,
replacing the drawings. Nothing else needs changing — the page looks for these
exact names and quietly falls back to a drawing when a file is not there.

```
static/looks/
  japandi-1.jpg   japandi-2.jpg   japandi-3.jpg
  scandinavian-1.jpg  scandinavian-2.jpg  scandinavian-3.jpg
  mid-century-modern-1.jpg  …
  industrial-1.jpg  …
  bohemian-1.jpg  …
  modern-luxury-1.jpg  …
```

- **Names**: `<style-id>-1.jpg`, `-2`, `-3`. The style ids are the ones in
  `STYLES` in the page: `japandi`, `scandinavian`, `mid-century-modern`,
  `industrial`, `bohemian`, `modern-luxury`.
- **`.jpg` or `.png`**, either is fine — the page tries `.jpg` first.
- **Landscape**, roughly 4:3, around 1200px wide. Bigger just makes the page
  slow; the frames are small.
- **Three each** is what the popup shows. One or two is fine, the rest stay
  drawings.

Copy the same files into `docs/looks/` as well — that folder is what GitHub
Pages serves.

A photograph you do not have the rights to is not worth the trouble it causes.
Your own work, the client's own work, or something properly licensed.
