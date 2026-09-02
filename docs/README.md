# docs/ — the public pages

GitHub Pages serves this folder, which gives the project a real URL before any
server exists.

- `index.html` — the app. A copy of `static/showcase.html`.
- `plan.html` — the build plan for the client.

`index.html` is a copy, so re-copy it after changing the original:

    cp static/showcase.html docs/index.html

On Pages there is no server behind it, so the interface is fully browsable but
reading and drawing are off. Point it at a real server by adding `?server=` to
the URL:

    https://<user>.github.io/interior-design/?server=https://your-app.onrender.com

The address is remembered in that browser, so it only needs adding once.
