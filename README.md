# QPP-Benchmark
Evaluación comparativa (benchmark) de métodos Query Performance Prediction (QPP) para búsquedas Ad-hoc utilizando métricas de correlación.

## Ejecución del Script

Para ejecutar el script principal y realizar la evaluación de los métodos QPP, utilice el siguiente comando:

```bash
python -m EvaluacionQPP.main --dataset <nombre_del_dataset>
```
### Parámetros
- - `--dataset`: Identificador del dataset a utilizar. Los datasets disponibles son:
  - `antique_test`
  - `iquique_small`

Ejemplo de uso:
```bash
python -m EvaluacionQPP.main --dataset antique_test
```

# To-do

### Objetivos específicos

- [x] Revisar la literatura sobre métodos de QPP en búsquedas Ad-hoc sin el uso de inteligencia artificial
     - [x] Identificar y describir los principales métodos QPP utilizados en la actualidad
     - [x] Selección de 5 métodos QPP según relevancia: NQC, IDF, Clarity, WIG y UEF
- [x] Identificar y analizar los procesos estándar de evaluación aplicados a los métodos de QPP
     - [x] Explorar los procesos estándar de evaluación: ir_datasets, ir_measures y PyTerrier
     - [x] Determinar los datasets a utilizar (ir_datasets): BEIR, Cranfield, MS MARCO, Antique y CAR
     - [x] Configurar el entorno experimental
- [x] Implementar métodos QPP en búsquedas Ad-hoc sin inteligencia artificial para su evaluación utilizando métricas estandarizadas
     - [x] Implementación de método pre-retrieval: IDF
     - [x] Implementación de métodos post-retrieval: Clarity, WIG, NQC y UEF.
- [ ] Evaluar los resultados obtenidos de los métodos QPP implementados, determinando su efectividad en función a los resultados descritos en el estado del arte
- [ ] Analizar y documentar el rendimiento de los métodos QPP implementados para establecer una línea base para futuras comparaciones con nuevos enfoques
