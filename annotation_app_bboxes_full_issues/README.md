# Full-Issue Bounding-Box Annotation App

Minimal local browser app for annotating face bounding boxes on issue PDFs.

## Features

- Load a local PDF in the browser
- Render one page at a time
- Navigate pages with the left and right arrow keys
- Draw multiple bounding boxes per page
- Hold `Ctrl` while drawing to save `analyzable = false`
- Right click while drawing to abort the current box
- Autosave annotations locally and reload them when the same PDF filename is opened again
- Export CSV with normalized coordinates and `issue_page_identifier`

## Run

```powershell
cd annotation_app_bboxes_full_issues
npm install
npm start
```

Then open the URL printed in the terminal. It starts at `http://localhost:5175` and will move to the next free port if needed.

## CSV columns

- `issue_page_identifier`
- `x1_rel`
- `y1_rel`
- `x2_rel`
- `y2_rel`
- `analyzable`

For a filename containing a date such as `ECON-1980-0802.pdf`,
`issue_page_identifier` is saved as `1980-0802-0045`, where `0045` is the
zero-based PDF page number padded to four digits. For other filenames, the
filename stem is used instead.

## Saved data

Autosaves are stored as `data/<pdf-filename>_annotations.csv`. The app only
loads files at that location and does not scan subdirectories. Annotation CSVs
and source PDFs are intentionally ignored by Git; supply a PDF locally for
testing.
