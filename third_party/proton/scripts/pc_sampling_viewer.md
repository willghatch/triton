# PC sampling HTML report

Generate a self-contained, interactive report with Python's standard library:

```sh
python third_party/proton/scripts/pc_sampling_viewer.py profile.hatchet -o report.html
```

Open `report.html` in a browser. No server, GPU, installed Triton package, or
Hatchet Python library is needed. The report loads its pinned Prism syntax
highlighter from jsDelivr; without network access, source remains readable as
unhighlighted text. The input must be a **raw Proton Hatchet JSON export
collected with PC sampling**. MessagePack, extended traces, and Hatchet-derived
inclusive metrics are not supported by this report generator.

The context with the most samples is selected initially. The context selector
retains the full tree path and a unique tree ID, so identical kernel names stay
separate. Repeated launches already combined by the exporter remain aggregates.

## Views

- **Annotated source:** complete source files in original line order, with
  count bars and context shares beside every line. Select a line number for its
  reason breakdown and local stalled fraction. Multiple functions on one line
  are combined within this source view. Without matching source text, the view
  falls back to ranked locations.
- **Ranked hotspots:** locations sorted by the selected metric, including an
  unattributed-source row. The default is stalled samples when nonzero and
  available for every location, otherwise all samples. Reason buttons filter and
  re-rank. Stalled bars are segmented only when every location's reason counters
  are present and sum to its stalled count. An explicit normalized-composition
  mode is available for those bars; counts and context shares remain visible.
- **Stall reason heatmap:** locations × nonzero reason counters, with all-sample
  count bars alongside. Click a column heading to sort by its reason, or a row
  to open source. Choose a common global count scale, share of all context
  samples, or row normalization (reason count / all samples on the line).
  Missing counters show `N/A`; zero is distinct. Reasons need not form a
  partition to appear in this view.
Python source and previews use Prism 1.29.0 loaded from version-pinned jsDelivr
URLs with subresource-integrity hashes. Whole-file tokenization preserves
multiline strings. If Prism cannot be loaded, the report falls back to plain
source text while keeping all views functional.

The shared metric selector offers all samples, stalled samples, and observed
reasons whose counters are complete in the selected context. Ranked-hotspot and
heatmap row selection opens annotated source when available, and always
populates the location details. Percentages include unattributed samples in the
denominator, even while displaying one source file.

## Source files

The generator embeds readable UTF-8 source files referenced by the profile.
Absolute paths are used directly; relative paths resolve from the current
directory, or from `--source-root`. To relocate paths captured on another
machine, use repeatable prefix mappings:

```sh
python third_party/proton/scripts/pc_sampling_viewer.py profile.hatchet \
  --source-root /work/project \
  --path-map /old/project=/work/project \
  --path-map /old/triton=/work/triton \
  -o report.html
```

Longest matching prefixes win and mappings respect path boundaries. Relative
mapping destinations also resolve against `--source-root`. Missing files do not
prevent report generation. Use `--no-source` for a report with only profile
locations and counters. Reports with sources include their full text and paths;
keep that in mind when sharing a report. Source content is embedded as escaped
JSON and displayed as text, including when it contains HTML or script tags.

## Counter interpretation and report limits

Raw Proton kernel counters are **unattributed samples**, not inclusive totals.
They are added to source-child counters exactly once. The report reads JSON
directly and does not request inclusive metrics from the Hatchet library.
Locations remain within their nearest non-source tree context; the tool does
not combine descendants across other kernel/context boundaries.

Counter presence determines availability. Missing counters are not filled with
zero. Zero-filled stall counters with no observed reasons are ambiguous: some
collection modes cannot observe stalls, and the export may not carry enough
metadata to distinguish those modes from a measured zero. The report calls out
that ambiguity and defaults to all samples. Backend reason names are retained;
no equivalence between NVIDIA and AMD categories is assumed.

Sample shares measure observations, not time or optimization savings. The
report shows source attribution coverage, but cannot verify source/build
identity, recover PCs or per-launch timelines, identify dropped observations,
or reconstruct unknown-kernel associations. Out-of-range source lines are
flagged and remain accessible in Ranked hotspots. Very large source files and
profiles are rendered directly without pagination or virtual scrolling.
