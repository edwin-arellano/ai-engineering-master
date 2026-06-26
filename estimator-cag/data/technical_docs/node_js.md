# Referencia técnica — Node.js

> Documento de referencia interno (corpus sembrado, S10). Material DESCRIPTIVO para
> apoyar el routing y la recuperación; **no contiene cifras de esfuerzo**. Las horas
> de estimación viven exclusivamente en los presupuestos históricos.

## Visión general

Node.js es la tecnología principal de varios proyectos del corpus histórico, en los
sectores: ecommerce, finance, healthcare. Esta referencia resume el stack y las integraciones técnicas
observadas para esa base tecnológica, como contexto de apoyo a la estimación.

## Integraciones típicas

Componentes técnicos de stack frecuentes en proyectos Node.js del corpus:

- elasticsearch
- node_js
- postgresql
- python
- redis

Capacidades funcionales recurrentes construidas sobre Node.js:

- Cart and checkout service
- Delivery slot booking
- Faceted search
- Fraud scoring engine
- Payment gateway adapter
- Product recommendation widget
- Route optimization
- Video consultation
- e-Prescription module

## Riesgos de estimación

Consideraciones cualitativas al estimar proyectos Node.js (sin cuantificar esfuerzo):

- La complejidad real depende del grado de integración con sistemas externos y del
  cumplimiento regulatorio del sector (ecommerce, finance, healthcare).
- Las capacidades transversales (autenticación, idempotencia, observabilidad) tienden
  a subestimarse cuando no aparecen como componentes explícitos.
- La cercanía de un proyecto nuevo a los históricos de esta tecnología es el mejor
  indicador de fiabilidad de la estimación por analogía.
