# How-it-works demo video

`../how-it-works-demo.mp4` — 16 s, 1280x720, no audio.

It is a screen recording of the real studio page. Only the paid calls are
stubbed, and the three room pictures are drawn stand-ins (`rooms.py`), so the
"before" and "after" are illustrations, not real AI output.

Re-record (after a rename, or with a real before/after pair):

1. Drop a real `before.jpg`, `after.jpg` (the redraw) and `edited.jpg` (the
   marked-area edit) next to these scripts, or run `python rooms.py` for stand-ins.
2. `python demo_server.py` (serves on :8767), then `python record.py`.
3. Trim with ffmpeg from the offset written to `t0.txt`, encode H.264 + `+faststart`.
