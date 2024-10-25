# QPP-Benchmark
Evaluación comparativa (benchmark) de métodos Query Performance Prediction (QPP) para búsquedas Ad-hoc utilizando métricas de correlación.

# To-do

### Objetivos específicos

- [x] Revisar la literatura sobre métodos de QPP en búsquedas Ad-hoc sin el uso de inteligencia artificial
     - [x] Identificar y describir los principales métodos QPP utilizados en la actualidad
     - [x] Selección de 5 métodos QPP según relevancia: NQC, IDF, Clarity, WIG y UEF
- [ ] Identificar y analizar los procesos estándar de evaluación aplicados a los métodos de QPP
     - [x] Explorar los procesos estándar de evaluación en TIREx: ir_datasets, ir_measures y PyTerrier
     - [x] Determinar los datasets a utilizar (ir_datasets): TREC Robust04, MS MARCO, ClueWeb09-12, TREC COVID y TREC Web Tracks
     - [ ] Configurar el entorno experimental en TIREx
     - [ ] Revisar y configurar métricas de correlación (ir_measures)
- [ ] Implementar métodos QPP en búsquedas Ad-hoc sin inteligencia artificial para su evaluación utilizando métricas estandarizadas
- [ ] Evaluar los resultados obtenidos de los métodos QPP implementados, determinando su efectividad en función a los resultados descritos en el estado del arte
- [ ] Analizar y documentar el rendimiento de los métodos QPP implementados para establecer una línea base para futuras comparaciones con nuevos enfoques
