# Máster AI Engineering — Entregables

Repositorio con los entregables del Máster AI Engineering de LIDR.

Cada sesión tiene dos ramas: `pre-session-NN` (entregable previo a la
sesión en vivo) y `session-NN` (estado tras la sesión en vivo).

> La Sesión 1 (notebook Jupyter con primeras llamadas a APIs de LLM) se
> omite de este repositorio porque su entregable es un notebook en
> Google Colab, no un proyecto Python.

## Estructura

- `estimator-cag/`: servicio FastAPI que construye un estimador de
  software basado en transcripciones de reunión. Crece a lo largo de
  todas las sesiones (CAG → wrapper → formulario → structured outputs
  → guardrails → cache semántico).

## Ramas

| Rama | Contenido |
|---|---|
| `pre-session-02` | Scaffolding FastAPI con arquitectura CAG (`POST /api/v1/estimate`) |

## Cómo arrancar

Cada subproyecto tiene su propio README con instrucciones específicas.
Empieza por `estimator-cag/README.md`.
