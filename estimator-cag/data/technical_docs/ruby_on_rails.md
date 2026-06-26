# Referencia técnica — Ruby on Rails

> Documento de referencia interno (corpus sembrado, S10). Material DESCRIPTIVO para
> apoyar el routing y la recuperación; **no contiene cifras de esfuerzo**. Las horas
> de estimación viven exclusivamente en los presupuestos históricos.

## Visión general

Ruby on Rails es la tecnología principal de varios proyectos del corpus histórico, en los
sectores: ecommerce, finance, other. Esta referencia resume el stack y las integraciones técnicas
observadas para esa base tecnológica, como contexto de apoyo a la estimación.

## Integraciones típicas

Componentes técnicos de stack frecuentes en proyectos Ruby on Rails del corpus:

- postgresql
- python
- redis
- ruby_on_rails

Capacidades funcionales recurrentes construidas sobre Ruby on Rails:

- Adaptive path engine
- Course delivery
- OAuth 2.0 authentication backend
- PSD2 compliance module
- Split payment engine
- Transaction ledger service
- Vendor onboarding

## Riesgos de estimación

Consideraciones cualitativas al estimar proyectos Ruby on Rails (sin cuantificar esfuerzo):

- La complejidad real depende del grado de integración con sistemas externos y del
  cumplimiento regulatorio del sector (ecommerce, finance, other).
- Las capacidades transversales (autenticación, idempotencia, observabilidad) tienden
  a subestimarse cuando no aparecen como componentes explícitos.
- La cercanía de un proyecto nuevo a los históricos de esta tecnología es el mejor
  indicador de fiabilidad de la estimación por analogía.
