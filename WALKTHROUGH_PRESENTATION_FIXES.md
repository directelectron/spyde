# Walkthrough + presentation/report fix batch

User-reported bugs from a walkthrough screenshot + presentation testing. Branch: feat/docs-walkthroughs.

## A. Walkthrough system

### A1. Move "Tutorial Data" → Examples → "Dummy Data"
- `MenuBar.tsx:79-83`: remove the standalone `'Tutorial Data'` top-level menu. Fold the 7 `TUTORIAL_DATA` items into the **Examples** dropdown as a "Dummy Data" group (MenuBar has no nested-submenu support — add a separator + labeled group, or extend Item for a submenu). Keep `testId: tutorial-<key>` + `sendAction('tutorial_load',{name:key})`.
- Optional parity: `electron/src/main/index.ts:538-551` native Examples submenu — add Dummy Data there too.

### A2. Drop "Show me" auto-drive entirely (USER DECISION: remove interactivity, just show steps)
This also FIXES the double/triple-load (copies #2 and #3 came from the step `drive`s):
- `guides/types.ts`: remove `drive`, `autoDrive` from `GuideStep`; remove `GuideDrive` usage on steps. KEEP `Guide.autoload` (the one legit load on tour open). Actually — reconsider: `GuideDrive` type is still used by `autoload`. Keep the type; just stop steps from carrying `drive`/`autoDrive`.
- All `guides/*.ts`: strip `drive`/`autoDrive` from every step (keep `anchor`/`title`/`body`/`placement`/`image`). Keep each guide's `autoload`.
- `Tour.tsx`: remove the "Show me ▶" button + `showMe()` + `runDrive` import + running/error states (105-119, 148, 225-238). Keep Back/Next/Done + the autoload effect (88-97) + spotlight.
- `guideDriver.ts`: the `runDrive` step-driver is now only used by `autoload`. Trim to just what `autoload` needs (a `backend` action + waitFor), or keep as-is if `autoload` reuses it. Playwright screenshot spec that drove steps: update/skip (screenshots can start from a tutorial_load).

### A3. Plot-toolbar highlight invisible
- `Tour.tsx:21-30` `resolveAnchor`: a hidden-but-laid-out target (opacity:0) returns a valid rect → spotlight over an invisible bar. Add an opacity/visibility check (mirror `guideDriver.ts:54-60 isVisible`) → treat as not-found (center the bubble) OR skip the spotlight.
- Better for the `floating-toolbar` step specifically: it's only visible on hover. Options: (a) re-anchor the step to the subwindow titlebar (always visible), or (b) since we're removing auto-drive, just soften the step so it doesn't demand a hidden anchor. Simplest: make `resolveAnchor` opacity-aware so a hidden anchor gracefully centers instead of drawing a phantom box.

### A4. Close the tutorial dataset when the walkthrough ends
- `Tour.tsx`: on `onClose`/Done, dispatch a backend close of the dataset(s) the tour's autoload opened. Need a backend action to close a tree by source_path (or close the most-recent tutorial tree). Check existing close/remove actions in spyde/backend.
- Track what autoload opened so we only close that, not a user's own data. Simplest robust approach: close trees whose source_path starts with "tutorial_".

### (Root-cause note) double-load
- `session.py:302-313` `_add_signal` appends unconditionally, no dedup by source_path. A2 removes the step-drive dupes. If autoload itself can double-fire (StrictMode / re-mount), also guard the Tour autoload effect to run once. Consider a backend dedup: tutorial_load focuses an existing tutorial_<name> tree instead of appending a duplicate.

## B. Presentation / report

### B1. Split figure+text: text not rendering in Present mode
- `PresentMode.tsx` `SlideCell` (601-618) and `PreviewCell` (505-533): add a `cell.cell_type === 'split'` branch that renders BOTH `renderMarkdown(cell.source)` and the figure side, honoring `split_layout` (text-left/text-right). Backend already sends `source` + `figure` for splits. Sidebar (`ReportSplitCell.tsx`) is fine.

### B2. Combo figures on slides (+ split figure side) — no compose drop path
- Backend `compose.py` already supports compose on any figure-like cell (incl. split-with-spec), no doc_type gate.
- `PresentMode.tsx` `SlideFigure` (649-678): mount the same compose shield / drop zones as `ReportFigureCell` so dropping a window pill onto a slide figure combines panels. (Presentations are edited in the sidebar, not Present mode — VERIFY where slide figures are actually edited: the sidebar renders slides too when doc_type==='presentation'. The real gap may be that a presentation's figure cells in the SIDEBAR already use ReportFigureCell and DO support compose — need to confirm the actual broken surface before wiring Present mode, which is a read-only viewer.)
- Split figure side (`ReportSplitCell.tsx`): once filled, wire compose zones like ReportFigureCell so you can add panels to a split's figure.
- **VERIFY FIRST**: reproduce "combo on slides is broken" — is the slide authored in the sidebar (ReportFigureCell, should work) or is the user trying to drop in Present mode (read-only)? Confirm the actual broken path before building.

### B3. Windows: overview/exit buttons covered by title bar
- `PresentMode.tsx:733-736` `topBar` `top:16` sits under the 38px Windows `titleBarOverlay` + native WCO (top-right). Push `top` to ≥ 44 on Windows (or move the bar so it clears the WCO region). SlideOverview overlay (193-198) similar — check its header ✕.

### B4. Speaker/presenter icon is bad
- `PresentMode.tsx:323` uses emoji `🗣`. Replace with a proper icon (an SVG, matching the app's icon style). Overview button `▦` (308) and exit `✕` are fine-ish; focus on the speaker glyph.

### B5. Figure cell edit/add/copy toolbar "small and ugly"
- The `CellChrome` hover pill: 13px glyphs, ~2px padding (`ReportFigureCell.tsx:1977-1988` chrome/chromeBtn; `ReportSplitCell.tsx:367-379`; `CellChrome.tsx` chromeBtn). Make it bigger/cleaner: larger hit targets, clearer icons, better background/spacing. Apply consistently across figure + split chrome.

## Verify
- Python: unaffected mostly (menu/tour/present are renderer). If adding a close-tutorial backend action, add a test.
- Build electron (`npm run build`), then drive the real app per CLAUDE.md: start a walkthrough → ONE dataset loads, steps show (no Show me), highlight not phantom, dataset closes at end. Presentation: split cell shows text+figure on a slide; combo-figure on a slide figure; Windows buttons not clipped; new speaker icon; nicer chrome. Screenshots reviewed.
- NEVER kill processes by name (user runs their own SpyDE; Playwright manages its own).
