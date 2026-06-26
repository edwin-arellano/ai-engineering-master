# Referencia técnica — Java

> Documento de referencia interno (corpus sembrado, S10). Material DESCRIPTIVO para
> apoyar el routing y la recuperación; **no contiene cifras de esfuerzo**. Las horas
> de estimación viven exclusivamente en los presupuestos históricos.

## Visión general

Java es la tecnología principal de varios proyectos del corpus histórico, en los
sectores: finance, healthcare, industrial. Esta referencia resume el stack y las integraciones técnicas
observadas para esa base tecnológica, como contexto de apoyo a la estimación.

## Integraciones típicas

Componentes técnicos de stack frecuentes en proyectos Java del corpus:

- java
- postgresql
- python

Capacidades funcionales recurrentes construidas sobre Java:

- Appointment scheduling
- Credit decisioning service
- Dynamic slotting
- EHR integration (HL7/FHIR)
- Loan origination workflow
- Warehouse management core

## Riesgos de estimación

Consideraciones cualitativas al estimar proyectos Java (sin cuantificar esfuerzo):

- La complejidad real depende del grado de integración con sistemas externos y del
  cumplimiento regulatorio del sector (finance, healthcare, industrial).
- Las capacidades transversales (autenticación, idempotencia, observabilidad) tienden
  a subestimarse cuando no aparecen como componentes explícitos.
- La cercanía de un proyecto nuevo a los históricos de esta tecnología es el mejor
  indicador de fiabilidad de la estimación por analogía.
