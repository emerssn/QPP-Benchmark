# EvaluacionQPP - Evaluación de Métodos de Predicción de Rendimiento de Consultas

## Descripción
Este proyecto implementa y evalúa diferentes métodos de Predicción de Rendimiento de Consultas (QPP) sobre colecciones de documentos. Los métodos QPP intentan predecir qué tan bien funcionará una consulta antes de ejecutarla.

## Características Principales
- ✨Implementación de métodos QPP pre y post-recuperación
- 📚Soporte para múltiples datasets (ANTIQUE, Iquique)
- 📊Análisis de correlación con métricas de evaluación (nDCG, AP)
- 📈Generación de gráficos y reportes de resultados
- ⚙️Procesamiento configurable de consultas y documentos

## Métodos QPP Implementados
### Pre-recuperación:
- IDF promedio y máximo
- SCQ promedio y máximo
- Clarity Score

### Post-recuperación:
- WIG (Weighted Information Gain)
- NQC (Normalized Query Commitment)
- UEF (Utility Estimation Framework)

## Requisitos
- Python 3.9+
- Java 11+
- Dependencias listadas en requirements.txt

## Instalación
1. Clonar el repositorio
2. Instalar Java 11
3. Crear entorno virtual: python -m venv qppenv
4. Activar entorno virtual:
   - Windows: qppenv\Scripts\activate
   - Unix/MacOS: source qppenv/bin/activate
5. Instalar dependencias: pip install -r requirements.txt

## Uso
Ejecutar evaluación completa:
python -m EvaluacionQPP.main --datasets antique_test iquique_small

### Opciones principales:
| Opción | Descripción |
|--------|-------------|
| `--datasets` | Datasets a evaluar (separados por espacios) |
| `--max-queries` | Número máximo de consultas a procesar |
| `--list-size` | Tamaño de lista para métricas de ranking |
| `--metrics` | Métricas de evaluación a utilizar |
| `--correlations` | Coeficientes de correlación a calcular |
| `--output-dir` | Directorio para guardar resultados |
| `--use-uef` | Incluir método UEF en evaluación |
| `--skip-plots` | Omitir generación de gráficos |

## Docker
También se puede ejecutar usando Docker:
1. Construir imagen: docker build -t qpp-eval .
2. Ejecutar contenedor: docker run qpp-eval

## Estructura del Proyecto
/EvaluacionQPP
  /data - Gestión de datasets
  /indexing - Construcción de índices
  /metodos - Implementación de métodos QPP
  /retrieval - Funciones de recuperación
  /utils - Utilidades generales
  /correlation_analysis - Análisis de resultados

## Contribuciones
Las contribuciones son bienvenidas. Por favor, seguir las guías de estilo:
- Usar snake_case para variables y funciones
- Usar camelCase para clases
- Usar ALL_CAPS para constantes
- Seguir principios OOP
- Mantener código modular y reutilizable
- Incluir documentación y comentarios
