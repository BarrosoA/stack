# TheBox

Minimalist static document reader for GitHub Pages across desktop and mobile devices.

## Directory Structure

```text
TheBox/
├── .github/
│   └── workflows/
│       └── deploy.yml          # automated manifest sync and github pages deployment
├── assets/
│   ├── css/
│   │   └── style.css           # dark minimalist layout and responsive styles
│   └── js/
│       └── app.js              # library view and lazy canvas pdf/text reader
├── documents/
│   ├── gradient-descent-for-machine-learning.pdf
│   └── manifest.json           # document metadata index
├── scripts/
│   └── sync-manifest.js        # local and ci manifest generator
├── .gitignore
├── .nojekyll
├── index.html
└── README.md
```

## Adding Documents

1. Place `.pdf`, `.md`, or `.txt` files inside `documents/`.
2. Run `node scripts/sync-manifest.js` locally (or push directly to GitHub; the workflow in `.github/workflows/deploy.yml` updates `documents/manifest.json` during deployment).
3. Optional custom titles or subtitles can be edited directly in `documents/manifest.json`.

## Controls

- `/` : Focus document filter in library view
- `Esc` : Return to library view
- `ArrowLeft` / `ArrowRight` : Previous or next page
- `+` / `-` / `0` : Zoom in, zoom out, or reset zoom
- `I` : Toggle dark page inversion

