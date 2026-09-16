#!/usr/bin/env python3
"""Generate a standalone PC sampling explorer from a raw Proton Hatchet JSON file."""

import argparse
import json
import re
from pathlib import Path

SOURCE_LOCATION = re.compile(r"^(.*):(\d+)@(.*)$")


def normalize_profile(profile):
    """Read raw (exclusive) counters, retaining each tree context independently.

    Kernel-level samples are unattributed, not inclusive source totals. A source
    frame belongs to its nearest non-source parent; same-named contexts are never
    merged. Hatchet-derived inclusive data is intentionally not supported.
    """
    roots = profile if isinstance(profile, list) else [profile]
    if not roots or not isinstance(roots[0], dict) or "frame" not in roots[0]:
        raise ValueError("Expected raw Proton Hatchet JSON with frame/metrics/children nodes")
    kernels = []

    def counters(node):
        metrics = node.get("metrics", {})
        result = {}
        for key, value in metrics.items():
            if key in ("num_samples", "num_stalled_samples") or key.startswith("stalled_"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"Invalid nonnegative integer counter {key}: {value!r}")
                result[key] = value
        if any(result.values()) and "num_samples" not in result:
            raise ValueError("PC sampling counters require num_samples; use a raw Proton export")
        if result.get("num_stalled_samples", 0) > result.get("num_samples", 0):
            raise ValueError("num_stalled_samples exceeds num_samples")
        return result

    def walk(node, path, identity, owner=None):
        name = node["frame"]["name"]
        loc = SOURCE_LOCATION.fullmatch(name)
        values = counters(node)
        if not loc or owner is None:
            owner = {
                "id": identity, "name": name, "context": " / ".join(path + [name]), "metadata": node.get("metrics", {}),
                "rows": []
            }
            kernels.append(owner)
        if values.get("num_samples", 0):
            owner["rows"].append({
                "label": name if loc else "Unattributed source", "file": loc[1] if loc else None, "line":
                int(loc[2]) if loc else None, "function": loc[3] if loc else None, "metrics": values
            })
        for index, child in enumerate(node.get("children", [])):
            walk(child, path + [name], f"{identity}.{index}", owner)

    for index, root in enumerate(roots):
        if isinstance(root, dict) and "frame" in root:
            walk(root, [], str(index))
    kernels = [kernel for kernel in kernels if kernel["rows"]]
    if not kernels:
        raise ValueError("No PC samples found. Collect a profile with PC sampling enabled.")
    for kernel in kernels:
        # Duplicate source labels can occur in a raw tree. Combine only within
        # this context, and preserve missing fields as unavailable.
        grouped = {}
        for row in kernel["rows"]:
            key = (row["file"], row["line"], row["function"])
            if key not in grouped:
                grouped[key] = row
            else:
                old = grouped[key]["metrics"]
                grouped[key]["metrics"] = {k: old[k] + row["metrics"][k] for k in old.keys() & row["metrics"].keys()}
        kernel["rows"] = list(grouped.values())
        for index, row in enumerate(kernel["rows"]):
            row["id"] = index
    return {"kernels": kernels, "sources": {}, "source_paths": {}}


def attach_sources(data, source_root, mappings):
    """Embed only referenced source files; explicit prefix mappings win."""
    mappings = sorted(mappings, key=lambda item: len(item[0]), reverse=True)
    for filename in sorted({row["file"] for kernel in data["kernels"] for row in kernel["rows"] if row["file"]}):
        path = Path(filename)
        for old, new in mappings:
            if filename == old or filename.startswith(old.rstrip("/") + "/"):
                path = Path(new) / filename[len(old):].lstrip("/")
                break
        if not path.is_absolute():
            path = source_root / path
        try:
            data["sources"][filename] = path.read_text(encoding="utf-8")
            data["source_paths"][filename] = str(path.resolve())
        except (OSError, UnicodeError):
            pass  # Missing source is a supported ranked-view fallback.


def write_report(data, output):
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    # Never allow profile names or source code to terminate the JSON script tag.
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace(">",
                                                                                  "\\u003e").replace("&", "\\u0026")
    output.write_text(template.replace("/*PROFILE_DATA*/", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="The output is a self-contained HTML page with annotated source, ranked hotspots, "
        "and a stall reason heatmap. Open it directly in a browser; no web server is required.")
    parser.add_argument("profile", type=Path, help="Raw Proton .hatchet JSON file")
    parser.add_argument("-o", "--output", type=Path, default=Path("pc-sampling.html"), metavar="HTML_FILE",
                        help="Output HTML file path, not a directory (default: ./pc-sampling.html). "
                        "The parent directory must exist; an existing file is overwritten.")
    parser.add_argument("--source-root", type=Path, default=Path.cwd(),
                        help="Base directory for relative source paths (default: current directory)")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="Replace a source path prefix; repeat as needed (longest match wins)")
    parser.add_argument("--no-source", action="store_true", help="Do not embed local source files")
    args = parser.parse_args()
    mappings = []
    for mapping in args.path_map:
        old, separator, new = mapping.partition("=")
        if not separator or not old or not new:
            parser.error("--path-map requires nonempty OLD=NEW")
        mappings.append((old, new))
    try:
        data = normalize_profile(json.loads(args.profile.read_text(encoding="utf-8")))
        data["profile"] = args.profile.name
        if not args.no_source:
            attach_sources(data, args.source_root, mappings)
        write_report(data, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"pc-sampling-viewer: {error}\n")
    print(f"Wrote {args.output.resolve()} ({len(data['kernels'])} contexts, {len(data['sources'])} source files)")


if __name__ == "__main__":
    main()
