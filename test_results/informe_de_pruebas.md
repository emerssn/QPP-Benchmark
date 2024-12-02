# Informe de Pruebas Unitarias

## Resumen

- **Fecha de Ejecución:** 2024-12-02 03:59:24
- **Duración Total:** 0:00:11.265284
- **Total de Pruebas:** 50
- **Pruebas Exitosas:** 50
- **Pruebas Fallidas:** 0

```
██████████████████████████████████████████████████ 100.0%
```

## Detalles por Clase de Prueba

### TestQPPCorrelationAnalyzer

#### Estadísticas

- **Pruebas Ejecutadas:** 10
- **Pruebas Exitosas:** 10
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 8.25 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_align_qids` | 0.001 |
| ✅ | `test_align_qids_with_iquique_dataset` | 0.002 |
| ✅ | `test_calculate_correlations` | 0.056 |
| ✅ | `test_edge_cases` | 0.002 |
| ✅ | `test_generate_report` | 3.705 |
| ✅ | `test_initialization` | 0.000 |
| ✅ | `test_plot_correlation_heatmap` | 0.621 |
| ✅ | `test_plot_correlations_across_datasets` | 0.290 |
| ✅ | `test_plot_correlations_boxplot` | 0.588 |
| ✅ | `test_plot_scatter_plots` | 2.984 |

---

### TestEvaluator

#### Estadísticas

- **Pruebas Ejecutadas:** 6
- **Pruebas Exitosas:** 6
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.05 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_invalid_queries` | 0.008 |
| ✅ | `test_multiple_metrics` | 0.014 |
| ✅ | `test_perfect_ap` | 0.008 |
| ✅ | `test_perfect_ndcg` | 0.008 |
| ✅ | `test_reversed_ap` | 0.008 |
| ✅ | `test_reversed_ndcg` | 0.008 |

---

### TestClarity

#### Estadísticas

- **Pruebas Ejecutadas:** 8
- **Pruebas Exitosas:** 8
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.03 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_calculate_kl_divergence` | 0.000 |
| ✅ | `test_compute_score` | 0.005 |
| ✅ | `test_compute_scores_batch` | 0.009 |
| ✅ | `test_compute_term_frequencies` | 0.001 |
| ✅ | `test_edge_cases` | 0.009 |
| ✅ | `test_get_collection_probabilities` | 0.005 |
| ✅ | `test_stemming_consistency` | 0.004 |
| ✅ | `test_term_cf_presence` | 0.000 |

---

### TestNQC

#### Estadísticas

- **Pruebas Ejecutadas:** 6
- **Pruebas Exitosas:** 6
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.01 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_calc_corpus_score` | 0.001 |
| ✅ | `test_calc_nqc` | 0.000 |
| ✅ | `test_compute_score` | 0.002 |
| ✅ | `test_compute_scores_batch` | 0.003 |
| ✅ | `test_edge_cases` | 0.001 |
| ✅ | `test_init_scores_vec` | 0.001 |

---

### TestWIG

#### Estadísticas

- **Pruebas Ejecutadas:** 7
- **Pruebas Exitosas:** 7
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.01 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_calc_corpus_score` | 0.001 |
| ✅ | `test_calc_wig` | 0.000 |
| ✅ | `test_compute_score` | 0.002 |
| ✅ | `test_compute_scores_batch` | 0.002 |
| ✅ | `test_edge_cases` | 0.001 |
| ✅ | `test_get_term_cf` | 0.000 |
| ✅ | `test_init_scores_vec` | 0.001 |

---

### TestIDF

#### Estadísticas

- **Pruebas Ejecutadas:** 7
- **Pruebas Exitosas:** 7
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.01 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_compute_score_empty_query` | 0.000 |
| ✅ | `test_compute_score_invalid_method` | 0.001 |
| ✅ | `test_compute_score_multiple_terms` | 0.001 |
| ✅ | `test_compute_score_single_term` | 0.001 |
| ✅ | `test_compute_score_unknown_term` | 0.000 |
| ✅ | `test_compute_scores_batch` | 0.003 |
| ✅ | `test_get_term_df` | 0.000 |

---

### TestSCQ

#### Estadísticas

- **Pruebas Ejecutadas:** 6
- **Pruebas Exitosas:** 6
- **Pruebas Fallidas:** 0
- **Tiempo de Ejecución:** 0.01 segundos

#### Casos de Prueba

| Estado | Prueba | Tiempo (s) |
|:------:|--------|------------|
| ✅ | `test_compute_score_empty_query` | 0.000 |
| ✅ | `test_compute_score_invalid_method` | 0.000 |
| ✅ | `test_compute_score_multiple_terms` | 0.001 |
| ✅ | `test_compute_score_single_term` | 0.002 |
| ✅ | `test_compute_score_unknown_term` | 0.000 |
| ✅ | `test_compute_scores_batch` | 0.007 |

---

