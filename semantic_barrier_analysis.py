"""
Semantic Barrier Analysis for CAR Dataset

This script analyzes the "semantic barrier" phenomenon in the CAR dataset,
where BM25 fails to retrieve relevant documents due to lexical mismatch
between Wikipedia section headings and paragraph content.

Uses the original pipeline components for consistent preprocessing.
"""

import os
import sys
import json
from pathlib import Path
from urllib.parse import unquote

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Define paths
EVAL_RESULTS_DIR = script_dir / "EvaluacionQPP" / "evaluation_results"
DATASETS = ["antique_test", "cranfield", "trec_covid", "msmarco_dl20_judged", "car_v15_trec_y1_manual"]


def load_results(dataset_name: str) -> dict:
    """Load results.json for a dataset."""
    results_path = EVAL_RESULTS_DIR / dataset_name / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_floor_effect_stats(results: dict, metric: str = "ndcg@10") -> dict:
    """Calculate floor effect statistics for a dataset."""
    if results is None or metric not in results:
        return None
    
    per_query = results[metric]["per_query"]
    scores = list(per_query.values())
    n_queries = len(scores)
    
    if n_queries == 0:
        return None
    
    # Count queries in different ranges
    zero_count = sum(1 for s in scores if s == 0.0)
    low_count = sum(1 for s in scores if 0.0 < s <= 0.1)
    mid_low_count = sum(1 for s in scores if 0.1 < s <= 0.3)
    mid_count = sum(1 for s in scores if 0.3 < s <= 0.5)
    high_count = sum(1 for s in scores if s > 0.5)
    
    mean_score = sum(scores) / n_queries
    
    return {
        "n_queries": n_queries,
        "mean": mean_score,
        "zero_pct": zero_count / n_queries * 100,
        "low_pct": (zero_count + low_count) / n_queries * 100,  # ≤0.1
        "ranges": {
            "0": zero_count,
            "(0, 0.1]": low_count,
            "(0.1, 0.3]": mid_low_count,
            "(0.3, 0.5]": mid_count,
            "> 0.5": high_count
        }
    }


def find_semantic_gap_examples(results: dict, n_examples: int = 5) -> list:
    """Find examples of queries with semantic gap (nDCG=0 with complex structure)."""
    if results is None or "ndcg@10" not in results:
        return []
    
    per_query = results["ndcg@10"]["per_query"]
    
    # Find queries with nDCG=0 that have hierarchical structure (multiple /)
    zero_queries = []
    for qid, score in per_query.items():
        if score == 0.0:
            decoded_qid = unquote(qid)
            # Count hierarchy depth
            depth = decoded_qid.count("/")
            if depth >= 2:  # At least 2 levels of hierarchy
                zero_queries.append({
                    "qid": qid,
                    "decoded": decoded_qid,
                    "depth": depth,
                    "ndcg": score
                })
    
    # Sort by depth (more complex = more illustrative)
    zero_queries.sort(key=lambda x: x["depth"], reverse=True)
    
    return zero_queries[:n_examples]


def generate_typst_table(stats_by_dataset: dict) -> str:
    """Generate Typst table code for floor effect comparison."""
    lines = []
    lines.append("// Tabla generada por semantic_barrier_analysis.py")
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: 5,")
    lines.append("    inset: (x: 8pt, y: 6pt),")
    lines.append("    align: center + horizon,")
    lines.append("    table.header[*Dataset*][*Consultas*][*nDCG\\@10 = 0*][*nDCG\\@10 ≤ 0.1*][*Media*],")
    
    for dataset, stats in stats_by_dataset.items():
        if stats is None:
            continue
        display_name = {
            "antique_test": "Antique/Test",
            "cranfield": "Cranfield",
            "trec_covid": "TREC-COVID",
            "msmarco_dl20_judged": "MS MARCO",
            "car_v15_trec_y1_manual": "CAR"
        }.get(dataset, dataset)
        
        lines.append(f"    [{display_name}], [{stats['n_queries']}], [{stats['zero_pct']:.1f}%], [{stats['low_pct']:.1f}%], [{stats['mean']:.3f}],")
    
    lines.append("  ),")
    lines.append("  caption: [Distribución del efecto suelo (floor effect) por dataset. Se muestra el porcentaje de consultas con rendimiento nulo o muy bajo.]")
    lines.append(") <tabla_floor_effect>")
    
    return "\n".join(lines)


def generate_examples_typst(examples: list) -> str:
    """Generate Typst content for semantic gap examples."""
    lines = []
    lines.append("// Ejemplos de brecha semántica generados por semantic_barrier_analysis.py")
    lines.append("")
    lines.append("Para ilustrar concretamente la barrera semántica, se presentan ejemplos de consultas CAR con rendimiento nulo:")
    lines.append("")
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: 2,")
    lines.append("    inset: (x: 8pt, y: 6pt),")
    lines.append("    align: (left, center),")
    lines.append("    table.header[*Consulta (estructura jerárquica)*][*nDCG\\@10*],")
    
    for ex in examples[:4]:  # Max 4 examples
        # Clean up the decoded query for display
        decoded = ex["decoded"].replace("%20", " ")
        # Escape special characters for Typst
        decoded = decoded.replace("_", "\\_").replace("&", "\\&")
        lines.append(f"    [{decoded}], [0.000],")
    
    lines.append("  ),")
    lines.append("  caption: [Ejemplos de consultas CAR con brecha semántica. Estas consultas derivan de encabezados jerárquicos de Wikipedia donde los términos de la consulta no coinciden léxicamente con los párrafos relevantes.]")
    lines.append(") <tabla_ejemplos_brecha_semantica>")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ANÁLISIS DE BARRERA SEMÁNTICA - DATASET CAR")
    print("=" * 60)
    
    # Calculate stats for all datasets
    stats_by_dataset = {}
    for dataset in DATASETS:
        results = load_results(dataset)
        stats = calculate_floor_effect_stats(results)
        stats_by_dataset[dataset] = stats
        
        if stats:
            print(f"\n{dataset}:")
            print(f"  Consultas: {stats['n_queries']}")
            print(f"  nDCG@10 = 0: {stats['zero_pct']:.1f}%")
            print(f"  nDCG@10 ≤ 0.1: {stats['low_pct']:.1f}%")
            print(f"  Media: {stats['mean']:.3f}")
    
    # Find semantic gap examples in CAR
    print("\n" + "=" * 60)
    print("EJEMPLOS DE BRECHA SEMÁNTICA EN CAR")
    print("=" * 60)
    
    car_results = load_results("car_v15_trec_y1_manual")
    examples = find_semantic_gap_examples(car_results, n_examples=10)
    
    for i, ex in enumerate(examples[:5], 1):
        print(f"\n{i}. {ex['decoded']}")
        print(f"   Profundidad jerárquica: {ex['depth']}")
        print(f"   nDCG@10: {ex['ndcg']}")
    
    # Generate Typst output
    print("\n" + "=" * 60)
    print("OUTPUT TYPST")
    print("=" * 60)
    
    typst_table = generate_typst_table(stats_by_dataset)
    typst_examples = generate_examples_typst(examples)
    
    # Save to file
    output_path = script_dir / "semantic_barrier_output.typ"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("// ===== TABLA DE FLOOR EFFECT =====\n\n")
        f.write(typst_table)
        f.write("\n\n")
        f.write("// ===== EJEMPLOS DE BRECHA SEMÁNTICA =====\n\n")
        f.write(typst_examples)
    
    print(f"\nOutput guardado en: {output_path}")
    
    # Print Typst content
    print("\n--- TABLA FLOOR EFFECT ---")
    print(typst_table)
    print("\n--- EJEMPLOS BRECHA SEMÁNTICA ---")
    print(typst_examples)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
