# Presentations

Talks about SpyDE, authored **in** SpyDE — each one is a real `.spyde-report`
presentation document, not a PDF.

| file | what it is | length |
|---|---|---|
| `spyde-overview.spyde-report` | "SpyDE — an overview": where the project came from, the HyperSpy stack it builds on, the Electron/anyplotlib architecture, the case for open and reproducible analysis, and where it is going | 21 slides, ~12.4 min |

## Opening a deck

1. Launch SpyDE (`cd electron && npm run dev`, or a released build).
2. Open the **Report** sidebar — the panel toggle at the top right of the app bar.
3. **Open**, and pick the `.spyde-report` file.
4. **Present** for the full-screen deck.

In Present mode: `→` / `Space` / `PageDown` advance, `←` / `PageUp` go back,
`Home` / `End` jump to the ends, `Esc` exits. **`S` toggles the presenter view** —
the current slide, the next one, the speaker notes, and a timer. A presentation
clicker sends arrow / page keys, so it works without extra setup.

Every slide carries speaker notes. They are visible only in the presenter view,
never to the audience and never in an exported deck.

## Editing

The deck is ordinary content — open it and edit the slides in the sidebar like
any report. Slides can be reordered by dragging their grips, and
`Export ▾` writes static HTML, interactive HTML, PDF, or a markdown folder.

`spyde-overview.spyde-report` is **generated**, so the durable source is the
script:

```bash
python doc/presentations/build_spyde_overview.py
```

Three tables in `build_spyde_overview.py` are what you edit:

| table | what it controls |
|---|---|
| `SPEAKER` | name, role, affiliation, email, **venue and date**. Feeds the title card, the closing card and the footer bar, so booking a new venue is one edit. Leave `venue`/`date` blank and the line is omitted rather than left dangling. |
| `THEME` | the deck's look — background, text, muted, accent, font stack, logo, footer. Written into the document's `theme:` front matter, so it travels with the file. |
| `SLIDES` | the talk: text, layout, speaker notes, per-slide time budget. |

The script prints the slide count, the total time budget and the resolved theme
on every rebuild, and **fails the build** if the budget leaves the ~11–13.5 min
slot the deck targets — better to find that here than on stage.

If you edit the deck in the app instead and save over the file, the script
becomes stale — that is fine, but say so in the commit.

Nothing on a slide is a placeholder. Anything still to be decided lives in the
speaker notes, where a projector can't show it to the room.

### Screenshots

The app screenshots live in `media/` and are captured from the **real app** by
`electron/tests/talk_screenshots.spec.ts` (a capture run, not a regression test):

```bash
cd electron
npx playwright test tests/talk_screenshots.spec.ts --project=electron \
  --reporter=line --retries=0
cp talk_shots/*.png ../doc/presentations/media/
```

`build_spyde_overview.py` crops each shot to the region that carries meaning
(`CROPS`) and caps its width (`IMAGE_WIDTH`) before embedding it, which is what
keeps the committed deck under a megabyte. Re-capturing at a different window
size will shift the crop boxes; the crop clamps rather than raising, so check the
slides afterwards.

### Verifying

`electron/tests/talk_present.spec.ts` opens the committed deck in the real app,
pages through every slide in Present mode, screenshots each one to
`electron/talk_present_shots/`, and fails on a slide that renders no text or
overflows horizontally. It also asserts the **theme** survives the round trip:
the deck's background and accent, the footer bar with its embedded logo and
contact line, that a title card carries *no* footer, and that slide headings
take the themed colour rather than the stylesheet's hard-coded one.

Look at the screenshots — that is the actual check. `N_SLIDES` and the theme
colours in the spec are duplicated from the build script on purpose: if you
change one, the other fails loudly.

```bash
cd electron
npx playwright test tests/talk_present.spec.ts --project=electron \
  --reporter=line --retries=0
```

## The file format

A `.spyde-report` is a plain zip you can unzip and read:

```
report.md          # YAML front-matter + markdown body — the whole document
figures/<id>.yaml  # a live figure recipe, per figure cell
assets/<id>.png    # the baked snapshot / embedded image, per cell
```

`report.md` is valid standalone markdown (pandoc-ready once unzipped).
Presentation-only attributes — slide breaks, title slides, background styles,
speaker notes — ride as invisible HTML comments, so an external markdown renderer
shows the prose and ignores the rest. `type: presentation` in the front matter is
what makes the document a deck rather than a scrolling report, and `theme:`
carries its look. The logo is embedded as a `data:` URL rather than a path, so
the deck survives being emailed to someone whose disk has never had this repo on
it.

This deck uses only markdown, image, and split cells, so it carries no live
signal bindings and opens standalone with no data loaded. A deck built from your
own session can instead hold **live figure cells** that re-bind to the signal when
you reopen it with the data loaded.
