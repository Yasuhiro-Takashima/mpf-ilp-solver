# MPF-ILP Solver

Maximum Priority Flow (MPF) Solver using Integer Linear Programming (ILP) with [Gurobi](https://www.gurobi.com/).

## Requirements

- Python 3.x
- [Gurobi](https://www.gurobi.com/) + `gurobipy`
- `networkx`

## Usage
```bash
# Run a specific method
python mpf_ilp.py <input.gpmax> --method <0-4>

# Run multiple methods
python mpf_ilp.py <input.gpmax> --method 1 3 4

# Run all methods (default)
python mpf_ilp.py <input.gpmax>
```

## Methods

| ID | Name | Description |
|----|------|-------------|
| 0 | MF | Maximum Flow (no priority) |
| 1 | MPF_bigM | MPF with big-M formulation |
| 2 | MPF_ind | MPF with indicator constraints |
| 3 | MPF_bigM_mstep | MPF big-M with multi-step solving |
| 4 | MPF_ind_mstep | MPF indicator with multi-step solving |

## Input Format

Files use the `gpmax` format:
```
p gpmax <nodes> <arcs> <po_relations>
n <node_id> <type>        (type: s=source, t=sink, or intermediate)
a <from> <to> <capacity>
po <arc_id1> <arc_id2>    (arc_id1 has higher priority than arc_id2)
```

## Benchmarks

Benchmark instances are located in the `benchmarks/` directory.

| File | Description |
|------|-------------|
| graph3x3.gpmax | 3x3 grid graph |
| graph5x5.gpmax | 5x5 grid graph |
| super_res-E1.gpmax | Super resolution instance E1 |
| super_res-E2.gpmax | Super resolution instance E2 |
| superres_graph.gpmax | Super resolution graph |
| texture-Cremer.gpmax | Texture synthesis (Cremer) |
| texture-Paper1.gpmax | Texture synthesis (Paper1) |
| texture-Temp.gpmax | Texture synthesis (Temp) |

## License

MIT License
