"""Escenarios sintéticos multi-turno para el stress test.

Cada escenario fuerza un comportamiento distinto del CAG:
- growing: requisitos coherentes acumulándose; mide coste y supervivencia del
  project_name a medida que crece N (es el perfil de la curva larga, 20 turnos).
- pivot: el turno 5 cambia el stack (React Native → Flutter); mide si el sistema
  ACTUALIZA el stack recordado o se queda anclado al antiguo.
- contradiction: el turno 3 fija el presupuesto en 30000, el turno 8 lo cambia a
  80000; mide cuál sobrevive en summary/anchors/metadata.

Decisión sobre `fact_to_remember` (ver notas del prompt, Paso 10): la
`MemoryDriftMetric` hace match LITERAL case-insensitive, así que los facts son
TÉRMINOS CORTOS buscables ("Nimbus", "Flutter", "30000"), no oraciones. Un fact
como "el presupuesto está bloqueado en 30000 euros" casi nunca aparecería
literal en el summary; "30000" sí es un proxy correcto de "el sistema recuerda
el presupuesto".

El runner (Bloque 5) evalúa, en cada turno N, si el `fact_to_remember` de ese
turno está presente en el snapshot del turno. En `growing` el fact es siempre
"Nimbus" para trazar directamente la curva recall-vs-N del nombre del proyecto;
en `pivot`/`contradiction` el fact sigue el término que importa en cada fase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    turn_index: int
    transcript: str
    fact_to_remember: str  # término corto buscado en summary/anchors/metadata


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[Turn]


GROWING = Scenario(
    name="growing",
    turns=[
        Turn(1, "We are building a B2B SaaS called Nimbus for sales teams. Contact management and a deal pipeline. Stack React + Postgres, team of 3.", "Nimbus"),
        Turn(2, "Add authentication with email + Google SSO and role-based access.", "Nimbus"),
        Turn(3, "Add a multi-tenant model so each customer org is isolated.", "Nimbus"),
        Turn(4, "Add an audit log of every change for compliance.", "Nimbus"),
        Turn(5, "Add CSV export of pipeline data with a daily scheduled job.", "Nimbus"),
        Turn(6, "Add a basic reporting dashboard with weekly email digests.", "Nimbus"),
        Turn(7, "Add a public REST API with rate limiting for integrations.", "Nimbus"),
        Turn(8, "Add webhooks so customers get notified of deal stage changes.", "Nimbus"),
        Turn(9, "Add a mobile-friendly responsive UI.", "Nimbus"),
        Turn(10, "Confirm the full scope so far and give a consolidated estimate.", "Nimbus"),
        Turn(11, "Add full-text search across contacts and deals.", "Nimbus"),
        Turn(12, "Add bulk import of contacts from CSV and vCard files.", "Nimbus"),
        Turn(13, "Add a Kanban board view for the deal pipeline.", "Nimbus"),
        Turn(14, "Add email templates and a mail-merge feature for outreach.", "Nimbus"),
        Turn(15, "Add a notifications center with in-app and email alerts.", "Nimbus"),
        Turn(16, "Add two-factor authentication via authenticator apps.", "Nimbus"),
        Turn(17, "Add a billing module with Stripe subscriptions and invoices.", "Nimbus"),
        Turn(18, "Add an analytics page with conversion funnels per sales rep.", "Nimbus"),
        Turn(19, "Add a data retention policy with automated archival of old deals.", "Nimbus"),
        Turn(20, "Confirm the entire scope across all turns and give the final consolidated estimate.", "Nimbus"),
    ],
)

PIVOT = Scenario(
    name="pivot",
    turns=[
        Turn(1, "Mobile app called Aurora for booking fitness classes. Stack React Native.", "React Native"),
        Turn(2, "Add push notifications for class reminders.", "React Native"),
        Turn(3, "Add a calendar with instructor availability.", "React Native"),
        Turn(4, "Add in-app payments via Stripe.", "React Native"),
        Turn(5, "Actually, we are switching the stack from React Native to Flutter.", "Flutter"),
        Turn(6, "Re-confirm the estimate with Flutter as the chosen stack.", "Flutter"),
        Turn(7, "Add an offline mode so users can browse classes without signal.", "Flutter"),
        Turn(8, "Give the final estimate, Flutter is locked as the stack.", "Flutter"),
    ],
)

CONTRADICTION = Scenario(
    name="contradiction",
    turns=[
        Turn(1, "Internal tool called Atlas for inventory tracking.", "Atlas"),
        Turn(2, "It must integrate with our SAP system.", "SAP"),
        Turn(3, "The budget is locked at 30000 EUR.", "30000"),
        Turn(4, "Add barcode scanning support.", "30000"),
        Turn(5, "Add a stock-level alerting system.", "30000"),
        Turn(6, "Add multi-warehouse support.", "30000"),
        Turn(7, "Add a supplier portal.", "30000"),
        Turn(8, "Update: the budget is now 80000 EUR.", "80000"),
    ],
)

SCENARIOS = {"growing": GROWING, "pivot": PIVOT, "contradiction": CONTRADICTION}
