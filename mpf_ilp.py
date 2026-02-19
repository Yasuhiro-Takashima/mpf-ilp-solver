#!/usr/bin/env python3
"""
Maximum Priority Flow Solver with ILP
"""
import networkx as nx

class GPMaxParser:
    """
    gpmax統一形式のシンプルなパーサー
    """
    
    def __init__(self, filename=None):
        """
        Args:
             filename: gpmax形式の入力ファイル名。省略時はparse()を後から呼ぶ。
        """
        self.n_nodes = 0
        self.n_arcs = 0
        self.n_po_relations = 0
        
        self.nodes = {}  # node_id -> {type, ...}
        self.arcs = []   # [{id, from, to, capacity}, ...]
        self.partial_order = []  # [(arc_id1, arc_id2), ...]
        
        if filename:
            self.parse(filename)
    
    def parse(self, filename):
        """
        gpmax形式のファイルをパース
        
        Args:
            filename: 入力ファイル名
        """
        arc_counter = 1  # 暗黙的arc ID
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # コメント行と空行をスキップ
                if not line or line.startswith('c'):
                    continue
                
                # コメント部分を除去
                if 'c' in line:
                    line = line.split('c')[0].strip()
                
                parts = line.split()
                if not parts:
                    continue
                
                # 行タイプで分岐
                if parts[0] == 'p':
                    self._parse_problem_line(parts)
                elif parts[0] == 'n':
                    self._parse_node_line(parts)
                elif parts[0] == 'a':
                    self._parse_arc_line(parts, arc_counter)
                    arc_counter += 1
                elif parts[0] == 'po':
                    self._parse_partial_order_line(parts)
    
    def _parse_problem_line(self, parts):
        """
        問題定義行をパース
        p gpmax <nodes> <arcs> <po_relations>
        """
        if len(parts) < 5:
            raise ValueError(f"Invalid problem line: {' '.join(parts)}")
        
        if parts[1] != 'gpmax':
            raise ValueError(f"Expected 'gpmax', got '{parts[1]}'")
        
        self.n_nodes = int(parts[2])
        self.n_arcs = int(parts[3])
        self.n_po_relations = int(parts[4])
    
    def _parse_node_line(self, parts):
        """
        ノード定義行をパース
        n <node_id> <type> [capacity]
        """
        if len(parts) < 3:
            raise ValueError(f"Invalid node line: {' '.join(parts)}")
        
        node_id = int(parts[1])
        node_type = parts[2]
        capacity = float(parts[3]) if len(parts) > 3 else float('inf')
        
        self.nodes[node_id] = {
            'type': node_type,
            'capacity': capacity
        }
    
    def _parse_arc_line(self, parts, arc_id):
        """
        枝定義行をパース
        a <from> <to> <capacity>
        
        Args:
            parts: 行の分割結果
            arc_id: 暗黙的arc ID（定義順序）
        """
        if len(parts) < 4:
            raise ValueError(f"Invalid arc line: {' '.join(parts)}")
        
        from_node = int(parts[1])
        to_node = int(parts[2])
        capacity = float(parts[3])
        
        self.arcs.append({
            'id': arc_id,
            'from': from_node,
            'to': to_node,
            'capacity': capacity
        })
    
    def _parse_partial_order_line(self, parts):
        """
        部分順序定義行をパース
        po <arc_id1> <arc_id2>
        
        意味: arc_id1 ≺ arc_id2
        """
        if len(parts) < 3:
            raise ValueError(f"Invalid partial order line: {' '.join(parts)}")
        
        arc1 = int(parts[1])
        arc2 = int(parts[2])
        
        self.partial_order.append((arc1, arc2))
    
    def get_arc_by_id(self, arc_id):
        """
        arc_idで枝を取得
        
        Args:
            arc_id: 枝ID（1-indexed）
        
        Returns:
            arc辞書 or None
        """
        for arc in self.arcs:
            if arc['id'] == arc_id:
                return arc
        return None
    
    def get_outgoing_arcs(self, node_id):
        """
        指定ノードから出る枝のリスト
        
        Args:
            node_id: ノードID
        
        Returns:
            枝のリスト
        """
        return [arc for arc in self.arcs if arc['from'] == node_id]
    
    def get_incoming_arcs(self, node_id):
        """
        指定ノードへ入る枝のリスト
        
        Args:
            node_id: ノードID
        
        Returns:
            枝のリスト
        """
        return [arc for arc in self.arcs if arc['to'] == node_id]
    
    def get_source_nodes(self):
        """ソースノードのリスト"""
        return [nid for nid, data in self.nodes.items() if data['type'] == 's']
    
    def get_sink_nodes(self):
        """シンクノードのリスト"""
        return [nid for nid, data in self.nodes.items() if data['type'] == 't']
    
    def validate(self):
        """
        基本的な検証
        
        Returns:
            (valid, errors): (bool, list of error messages)
        """
        errors = []
        
        # ノード数チェック
        if len(self.nodes) != self.n_nodes:
            errors.append(f"Node count mismatch: expected {self.n_nodes}, got {len(self.nodes)}")
        
        # 枝数チェック
        if len(self.arcs) != self.n_arcs:
            errors.append(f"Arc count mismatch: expected {self.n_arcs}, got {len(self.arcs)}")
        
        # 部分順序数チェック
        if len(self.partial_order) != self.n_po_relations:
            errors.append(f"PO relation count mismatch: expected {self.n_po_relations}, got {len(self.partial_order)}")
        
        # ノードIDが1以上か
        for node_id in self.nodes:
            if node_id < 1:
                errors.append(f"Invalid node ID: {node_id} (must be >= 1)")
        
        # 枝の参照する全ノードが定義されているか
        for arc in self.arcs:
            if arc['from'] not in self.nodes:
                errors.append(f"Arc {arc['id']} references undefined source node: {arc['from']}")
            if arc['to'] not in self.nodes:
                errors.append(f"Arc {arc['id']} references undefined target node: {arc['to']}")
        
        # 部分順序の参照する全arc IDが有効か
        valid_arc_ids = {arc['id'] for arc in self.arcs}
        for arc1, arc2 in self.partial_order:
            if arc1 not in valid_arc_ids:
                errors.append(f"Partial order references invalid arc ID: {arc1}")
            if arc2 not in valid_arc_ids:
                errors.append(f"Partial order references invalid arc ID: {arc2}")
        
        # ソースとシンクの存在チェック
        sources = self.get_source_nodes()
        sinks = self.get_sink_nodes()
        
        if not sources:
            errors.append("No source node defined")
        if not sinks:
            errors.append("No sink node defined")
        
        return (len(errors) == 0, errors)
    
    def summary(self):
        """インスタンスの要約を返す"""
        sources = self.get_source_nodes()
        sinks = self.get_sink_nodes()
        
        return {
            'nodes': self.n_nodes,
            'arcs': self.n_arcs,
            'po_relations': self.n_po_relations,
            'sources': sources,
            'sinks': sinks,
            'intermediate_nodes': len(self.nodes) - len(sources) - len(sinks)
        }
    
    def __repr__(self):
        summary = self.summary()
        return (f"GPMaxParser(\n"
                f"  nodes={summary['nodes']}, "
                f"arcs={summary['arcs']}, "
                f"po_relations={summary['po_relations']}\n"
                f"  sources={summary['sources']}, "
                f"sinks={summary['sinks']}\n"
                f")")

import gurobipy as gp
from gurobipy import GRB

def my_callback(model, where):
    """
    Gurobi MIP 最適化のコールバック関数（早期終了制御）

    経過時間が model._time_limit_cb を超えるか、実行可能解が見つかった場合に
    MIPギャップが十分小さければ最適化を打ち切る。
    model._threshold: 現在の終了ギャップ閾値（初期値 1.0）
    """
    if where == GRB.Callback.MIP:
        run_time = model.cbGet(GRB.Callback.RUNTIME)
        if model._threshold < 1.0 or run_time >= model._time_limit_cb:
            obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
            if obj_bst > 1e-10:
                obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                gap = (obj_bnd - obj_bst) / obj_bst
                if model._threshold > gap:
                    model._threshold = gap / 2.0
                    model.terminate()

METHOD_NAMES = ['MF', 'MPF_bigM', 'MPF_ind', 'MPF_bigM_mstep', 'MPF_ind_mstep']            
    
class ILP_Solver:
    """
    ILP Solver
    """
    def __init__(self, parser):
        """
        Args:
            parser: GPMaxParser インスタンス（パース済みであること）
        """
        self.out_arcs = {}
        self.in_arcs = {}
        self.capacities = {}
        self.dg = nx.DiGraph()

        self.initialize(parser)

    def initialize(self, parser):
        """
        パーサーの結果からソルバー内部データ構造を構築する。

        out_arcs, in_arcs: ノードID → 接続枝IDリスト のマッピング
        capacities: 枝ID → 容量 のマッピング
        dg: 部分順序関係を表す有向グラフ（枝IDをノードとする）
        """        

        # arc information
        for arc in parser.arcs:
            arc_id = arc['id']
            from_node = arc['from']
            to_node = arc['to']
            capacity = arc['capacity']
            self.out_arcs.setdefault(from_node, [])
            self.out_arcs[from_node].append(arc_id)
            self.in_arcs.setdefault(to_node, [])
            self.in_arcs[to_node].append(arc_id)
            self.capacities.setdefault(arc_id, capacity)
        # partial order information
        self.dg.add_nodes_from([arc['id'] for arc in parser.arcs])
        for (arc1, arc2) in parser.partial_order:
            self.dg.add_edge(arc1, arc2)
        self.source_set = set(parser.get_source_nodes())
        self.sink_set = set(parser.get_sink_nodes())
            
    
    def optimize(self, mpf_flag=1, time_limit_cb=300, flow_lb_tol=0.01, flow_ub_tol=0.99):
        """
        ILP モデルを構築して最適化を実行する
        
        Args: 
            mpf_flag (int): 最適化手法の選択
                0: 最大フロー（優先度制約なし）
                1: MPF big-M 定式化
                2: MPF indicator 制約定式化
                3: MPF big-M + 多段階求解
                4: MPF indicator + 多段階求解
            time_limit_cb (float): コールバックでの早期終了タイムリミット（秒, default: 300）
            flow_lb_tol (float): この値以下の flow を持つ枝を非使用と判断し rfu を 0 に固定する閾値（default: 0.01）
            flow_ub_tol (float): この値以上の flow を持つ枝を使用と判断し rfu を 1 に固定する閾値（default: 0.99）
        
        Returns:
            (total_flow, flow_val):
                total_flow (float): 目的関数値（最大優先フロー）
                flow_val (dict): {arc_id: flow値} フローが正の枝のみ
        """        

        # env
        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.setParam('DisplayInterval', 30)
        env.setParam('NodefileStart', 4)
        env.setParam('LogFile', 'mpf.log')
        env.setParam('TimeLimit', 3600)        
        env.start()
        # model initialization
        model = gp.Model(env=env, name=METHOD_NAMES[mpf_flag])
        model._threshold = 1.0
        model._time_limit_cb = time_limit_cb
        model._flow_lb_tol = flow_lb_tol
        model._flow_ub_tol = flow_ub_tol

        # variables
        # flow
        flow = {arc_id: model.addVar(vtype=GRB.CONTINUOUS, ub=1.0, name=f'f_{arc_id}') for arc_id in self.capacities.keys()}
        if mpf_flag >= 1:
            rfu = {arc_id: model.addVar(vtype=GRB.BINARY, name=f'rfu_{arc_id}') for arc_id in self.dg.nodes() if self.dg.in_degree(arc_id) > 0}
        
        # constraints
        # flow constraint
        for node in self.out_arcs.keys():
            if node in self.source_set or node in self.sink_set:
                continue
            out_arc_terms = gp.quicksum(self.capacities[arc_id] * flow[arc_id] for arc_id in self.out_arcs[node])
            in_arc_terms = gp.quicksum(self.capacities[arc_id] * flow[arc_id] for arc_id in self.in_arcs[node]) if node in self.in_arcs.keys() else 0
            model.addConstr(out_arc_terms - in_arc_terms == 0, f'flow_{node}')

        # priority
        if mpf_flag >= 1:
            for (p_arc, c_arc) in self.dg.edges():
                if mpf_flag % 2 == 1:
                    model.addConstr(flow[p_arc] - rfu[c_arc] >= 0, f'rfu_{c_arc}_{p_arc}')                    
                else:
                    model.addGenConstrIndicator(rfu[c_arc], True, flow[p_arc] >= 1.0, name=f'rfu_ind_{p_arc}_{c_arc}')                    
            for c_arc in self.dg.nodes():
                if self.dg.in_degree(c_arc) == 0:
                    continue
                model.addConstr(flow[c_arc] - rfu[c_arc] <= 0, f'cap_{c_arc}')                
                
        # objective
        # 目的関数: ソースから出る枝のフロー総量（容量×正規化フロー）を最大化
        model.setObjective(gp.quicksum(self.capacities[arc] * flow[arc] for s in self.source_set for arc in self.out_arcs[s]), GRB.MAXIMIZE)

        
        # optimize 
        # multi-step 求解: コールバックで早期終了後、rfu 変数の値を見て上下界を固定し再求解することで高速化する
        if mpf_flag >= 3:
            while 1:
                model.optimize(my_callback)
                if model.Status == GRB.OPTIMAL:
                    break
                else:
                    # flow値 がほぼ0 → 対応する枝は使わないと判断し ub=0 に固定
                    # flow値 がほぼ1 → 対応する枝は使うと判断し lb=1 に固定                    
                    flow_var = {arc: var.X for (arc, var) in flow.items()}
                    rfu_var =  {arc: var.X for (arc, var) in rfu.items()}
                    for (arc, val) in rfu_var.items():
                        if val <= 0.01:
                            for p_arc in self.dg.predecessors(arc):
                                if flow_var[p_arc] <= model._flow_lb_tol:
                                    rfu[arc].ub = 0
                        elif val >= 0.99:
                            if flow_var[arc] >= model._flow_ub_tol:
                                rfu[arc].lb = 1
        else:
            model.update()
            model.optimize()

        # solution
        if model.SolCount > 0:
            total_flow = model.objVal
            flow_val = {arc: var.X for (arc, var) in flow.items() if var.X > 0}
        return total_flow, flow_val
        
if __name__ == '__main__':
    import sys
    import time
    import argparse

    parser_arg = argparse.ArgumentParser(
        description='Maximum Priority Flow Solver using ILP'
    )
    parser_arg.add_argument('filename', help='Input file in gpmax format')
    parser_arg.add_argument(
        '--method', '-m',
        type=int,
        choices=range(len(METHOD_NAMES)),
        nargs='+',
        default=None,
        help=(
            'Optimization method: '
            '0=MF, 1=MPF_bigM, 2=MPF_ind, '
            '3=MPF_bigM_mstep, 4=MPF_ind_mstep '
            '(default: run all)'
        )
    )
    parser_arg.add_argument(
        '--time-limit-cb', '-t',
        type=float,
        default=300,
        help='Callback time limit in seconds for early termination (default: 300)'
    )
    parser_arg.add_argument(
        '--flow-lb-tol',
        type=float,
        default=0.01,
        help='Flow threshold below which rfu is fixed to 0 in multi-step solving (default: 0.01)'        
    )
    parser_arg.add_argument(
        '--flow-ub-tol',
        type=float,
        default=0.99,
        help='Flow threshold above which rfu is fixed to 1 in multi-step solving (default: 0.99)'
    )
    args = parser_arg.parse_args()
    filename = args.filename

    # 実行するメソッドのリストを決定
    methods_to_run = args.method if args.method is not None else list(range(len(METHOD_NAMES)))

    # Parse
    start_time = time.time()
    graph_parser = GPMaxParser(filename)
    parse_time = time.time()
    print("Parsing Time: %f"%(parse_time-start_time))
    
    # Modeling
    solver = ILP_Solver(graph_parser)
    initial_time = time.time()
    print("Initializing Time: %f"%(initial_time-parse_time))
        
    # Optimize
    for m in methods_to_run:
        opt_start_time = time.time()        
        (total_flow, flow_val) = solver.optimize(m, time_limit_cb=args.time_limit_cb, flow_lb_tol=args.flow_lb_tol, flow_ub_tol=args.flow_ub_tol)        
        opt_end_time = time.time()
        print(f"{METHOD_NAMES[m]}: {total_flow} {opt_end_time - opt_start_time}")
    
