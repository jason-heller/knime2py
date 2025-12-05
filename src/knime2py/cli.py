#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# knime2py.cli — KNIME → Python/Notebook codegen & graph exporter (CLI entry)
# -----------------------------------------------------------------------------

"""
KNIME workflow CLI parser and exporter.

Overview
----------------------------
This module parses a KNIME workflow and emits graph representations and 
workbooks for isolated subgraphs in Python or Jupyter Notebook formats.

Runtime Behavior
----------------------------
Inputs: The generated code reads a single KNIME workflow file or a directory 
containing it, specifically looking for 'workflow.knime'.

Outputs: The module writes output to specified directories, generating JSON 
and DOT graph files, as well as Python and Jupyter Notebook workbooks. The 
output includes mappings of nodes and edges.

Key algorithms or mappings: The code handles the parsing of workflow components 
and generates corresponding graph representations.

Edge Cases
----------------------------
The code handles cases where no nodes or edges are found in the workflow, 
and it raises appropriate errors for invalid paths.

Generated Code Dependencies
----------------------------
This module requires the following external libraries: pandas, numpy, 
sklearn, imblearn, matplotlib, and lxml. These dependencies are required 
by the generated code, not by this code.

Usage
----------------------------
This module is typically invoked from the command line, allowing users to 
specify the path to a KNIME workflow. For example, the context can be accessed 
as follows: `args.path`.

Node Identity
----------------------------
This module does not define specific KNIME factory IDs or special flags.

Configuration
----------------------------
This module does not parse a settings.xml file.

Limitations
----------------------------
This module does not support recursive searches for workflow files in 
directories and assumes a specific file structure.

References
----------------------------
Refer to the KNIME documentation for more details on workflow structures 
and node configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# NOTE: relative imports because we're now inside the package under src/
from .parse_knime import parse_workflow_components
from .emitters import (
    write_graph_json,
    write_graph_dot,
    write_workbook_py,
    write_workbook_ipynb,
    build_workbook_blocks,
)

def _resolve_metanode_paths(root_path: Path) -> list[Path]:
    """
    Returns all subdirectories containing metanodes
    
    :param path: The parent directory
    :type path: Path
    :return: A list of paths containing nodes
    :rtype: list[Path]
    """
    out_list = list({parent_path for parent_path in root_path.rglob('workflow.knime')})

    return out_list

def _resolve_single_workflow(path: Path) -> Path:
    """
    Return the path to a single workflow.knime based on the given path.

    Rules:
      - If 'path' is a file, it must be named 'workflow.knime'.
      - If 'path' is a directory, it must contain a file named 'workflow.knime' directly
        (no recursive search).

    Args:
        path (Path): The path to the workflow file or directory.

    Returns:
        Path: The resolved path to the workflow.knime file.

    Raises:
        SystemExit: If the path does not exist or is not a valid workflow file.
    """
    p = path.expanduser().resolve()

    if not p.exists():
        print(f"Path does not exist: {p}", file=sys.stderr)
        raise SystemExit(2)

    if p.is_file():
        if p.name != "workflow.knime":
            print(f"Not a workflow.knime file: {p}", file=sys.stderr)
            raise SystemExit(2)
        return p

    # Directory: only accept a workflow.knime directly inside it (no recursion)
    wf_path = p / "workflow.knime"
    if not wf_path.exists() or not wf_path.is_file():
        print(f"No workflow.knime found in directory: {p}", file=sys.stderr)
        raise SystemExit(2)
    return wf_path


def run_cli(argv: Optional[list[str]] = None) -> int:
    """
    Parse command-line arguments and execute the KNIME workflow parsing and exporting.

    Args:
        argv (Optional[list[str]]): The command-line arguments. If None, uses sys.argv.

    Returns:
        int: Exit code indicating success (0) or failure (non-zero).
    """
    p = argparse.ArgumentParser(
        description="Parse a single KNIME workflow and emit graph + workbook per isolated subgraph."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a workflow.knime file OR a directory that directly contains workflow.knime",
    )
    p.add_argument("--out", type=Path, default=Path("out_graphs"), help="Output directory")
    p.add_argument(
        "--workbook",
        choices=["py", "ipynb"],          # None => generate both
        default=None,
        help="Workbook format to generate. Omit to generate both.",
    )
    p.add_argument(
        "--graph",
        choices=["dot", "json", "off"],
        default=None,                     # None => generate both
        help="Which graph file(s) to emit: dot, json, or off. Omit to generate both.",
    )

    args = p.parse_args(argv)

    # TODO: Since we walk the directory for metanodes, we dont need to seek
    # for singular knime.workflow files anymore, this should be removed
    # as it creates redundancy with _resolve_metanode_paths
    wf_root_file = _resolve_single_workflow(args.path) 
    wf_root_path = _resolve_single_workflow(args.path).parent

    # All paths where nodes live
    wf_paths = _resolve_metanode_paths(wf_root_path)

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(wf_paths) == 0:
        print(f"No paths found to convert in {wf_root_path}")

    if len(wf_paths) > 1:
        metanode_path = out_dir.joinpath("metanodes")
        metanode_path.mkdir(parents=True, exist_ok=True)
        print(f"Creating metanode path {metanode_path}")

    for wf_path in wf_paths:
        is_root = wf_root_file == wf_path
        if is_root:
            print(f"Converting {wf_path.parent.name}")

        convert_path(args, wf_path, out_dir, is_root)
    
    return 0

def convert_path(args, wf_path: Path, out_dir, is_root):
    try:
        graphs = parse_workflow_components(wf_path)  # one WorkflowGraph per isolated component
    except Exception as e:
        print(f"ERROR parsing {wf_path}: {e}", file=sys.stderr)
        return 3

    # Okay for metanodes (not root) to be empty
    if not graphs and is_root:
        print(f"No nodes/edges found in workflow: {wf_path}", file=sys.stderr)
        return 4

    components = []
    for g in graphs:
        # Conditionally emit JSON/DOT based on --graph
        j = d = None
        if args.graph in (None, "json"):
            j = write_graph_json(g, out_dir)
        if args.graph in (None, "dot"):
            d = write_graph_dot(g, out_dir)
        # args.graph == "off" → skip both

        wb_py = wb_ipynb = None

        # Build blocks/imports once
        blocks, imports = build_workbook_blocks(g)

        # --- per-graph summaries
        idle_count = sum(1 for b in blocks if getattr(b, "state", None) == "IDLE")

        # Collect not-implemented node names with factories
        not_impl_names: set[str] = set()
        for b in blocks:
            if getattr(b, "not_implemented", False):
                node = getattr(g, "nodes", {}).get(getattr(b, "nid", None)) if hasattr(g, "nodes") else None
                factory = (
                    getattr(node, "type", None)
                    or getattr(node, "factory", None)
                    or "UNKNOWN"
                )
                title = getattr(b, "title", "UNKNOWN")
                not_impl_names.add(f"{title} ({factory})")

        # Workbooks
        wf_name = wf_path.parent.name
        if not is_root:
            wf_name = "metanodes/" + wf_name

        if args.workbook in (None, "py"):
            wb_py = write_workbook_py(g, wf_name, out_dir, blocks, imports)
        if args.workbook in (None, "ipynb"):
            wb_ipynb = write_workbook_ipynb(g, wf_name, out_dir, blocks, imports)

        components.append(
            {
                "workflow_id": getattr(g, "workflow_id", None),
                "json": str(j) if j else None,
                "dot": str(d) if d else None,
                "workbook_py": str(wb_py) if wb_py else None,
                "workbook_ipynb": str(wb_ipynb) if wb_ipynb else None,
                "nodes": len(getattr(g, "nodes", {})),
                "edges": len(getattr(g, "edges", [])),
                "idle": idle_count,
                "not_implemented_count": len(not_impl_names),
                "not_implemented_names": sorted(not_impl_names),
            }
        )

    summary = {
        "workflow": str(wf_path),
        "total_components": len(components),
        "components": components,
    }
    # print(json.dumps(summary, indent=2))

def main(argv: Optional[list[str]] = None) -> None:
    """
    Console entrypoint used by `pyproject.toml`.

    Args:
        argv (Optional[list[str]]): The command-line arguments. If None, uses sys.argv.
    """
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    # Support direct execution: python -m knime2py or python src/knime2py/cli.py
    main(sys.argv[1:])
