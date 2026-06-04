"""Genera data/budgets_sample.json: 15 presupuestos históricos sintéticos con
esquema anidado (client_metadata + components[]). Determinista y comiteado.
Variedad: finance / ecommerce / healthcare / industrial / other; stacks diversos.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "budgets_sample.json"


def _component(cid, name, desc, stack, hours, complexity, deps=()):
    return {
        "component_id": cid,
        "name": name,
        "description": desc,
        "tech_stack": list(stack),
        "estimated_hours": hours,
        "complexity": complexity,
        "dependencies": list(deps),
    }


def _budget(bid, name, sector, country, summary, tech, year, components):
    return {
        "budget_id": bid,
        "client_metadata": {"name": name, "sector": sector, "country": country},
        "project_summary": summary,
        "main_technology": tech,
        "year": year,
        "total_estimated_hours": sum(c["estimated_hours"] for c in components),
        "components": components,
    }


BUDGETS = [
    _budget("BUD-2024-001", "FintechCorp", "finance", "ES",
            "Mobile banking API with OAuth 2.0 authentication and PSD2 compliance", "ruby_on_rails", 2024, [
                _component("AUTH-001", "OAuth 2.0 authentication backend",
                           "OAuth 2.0 flows with JWT session management, multi-tenant token isolation and per-client rate limiting.",
                           ["ruby_on_rails", "postgresql", "redis"], 120, "high"),
                _component("PSD2-002", "PSD2 compliance module",
                           "Strong Customer Authentication and consent management aligned with PSD2 regulatory requirements.",
                           ["ruby_on_rails", "postgresql"], 90, "high", ["AUTH-001"]),
                _component("TXN-003", "Transaction ledger service",
                           "Double-entry ledger with idempotent transaction posting and reconciliation jobs.",
                           ["ruby_on_rails", "postgresql"], 110, "medium"),
            ]),
    _budget("BUD-2024-002", "PayFlow", "finance", "MX",
            "Payment gateway integration with fraud scoring", "node_js", 2024, [
                _component("PAY-001", "Payment gateway adapter",
                           "Adapters for multiple PSPs with unified charge/refund API and webhook reconciliation.",
                           ["node_js", "postgresql"], 100, "medium"),
                _component("FRAUD-002", "Fraud scoring engine",
                           "Rule-based plus heuristic fraud scoring on transaction streams with manual review queue.",
                           ["python", "redis"], 130, "high"),
            ]),
    _budget("BUD-2023-003", "LendEasy", "finance", "ES",
            "Loan origination platform with credit decisioning", "java", 2023, [
                _component("LOAN-001", "Loan origination workflow",
                           "Multi-step origination workflow with document upload and underwriting state machine.",
                           ["java", "postgresql"], 140, "high"),
                _component("CREDIT-002", "Credit decisioning service",
                           "Scorecard-based decisioning with explainability output for compliance review.",
                           ["python"], 95, "medium"),
            ]),
    _budget("BUD-2024-004", "InvestHub", "finance", "US",
            "Robo-advisor portfolio rebalancing service", "python", 2024, [
                _component("PORT-001", "Portfolio rebalancing engine",
                           "Periodic rebalancing against target allocations with tax-aware lot selection.",
                           ["python", "postgresql"], 120, "high"),
                _component("REPORT-002", "Client reporting module",
                           "Scheduled PDF statements and performance attribution dashboards.",
                           ["python", "react"], 70, "low"),
            ]),
    _budget("BUD-2024-005", "ShopNova", "ecommerce", "ES",
            "Headless e-commerce storefront with personalization", "node_js", 2024, [
                _component("CART-001", "Cart and checkout service",
                           "Server-side cart, multi-currency checkout and inventory reservation with timeout release.",
                           ["node_js", "postgresql", "redis"], 110, "medium"),
                _component("RECO-002", "Product recommendation widget",
                           "Collaborative-filtering recommendations served via low-latency edge cache.",
                           ["python", "redis"], 85, "medium"),
                _component("SEARCH-003", "Faceted search",
                           "Faceted catalog search with synonyms and typo tolerance.",
                           ["elasticsearch", "node_js"], 75, "low"),
            ]),
    _budget("BUD-2023-006", "MarketPlace24", "ecommerce", "BR",
            "Multi-vendor marketplace with split payments", "ruby_on_rails", 2023, [
                _component("VENDOR-001", "Vendor onboarding",
                           "KYC-lite vendor onboarding with payout account verification.",
                           ["ruby_on_rails", "postgresql"], 90, "medium"),
                _component("SPLIT-002", "Split payment engine",
                           "Order-level payment splitting across vendors with platform fee retention.",
                           ["ruby_on_rails"], 105, "high", ["VENDOR-001"]),
            ]),
    _budget("BUD-2024-007", "FreshCart", "ecommerce", "MX",
            "Grocery delivery slot booking and routing", "node_js", 2024, [
                _component("SLOT-001", "Delivery slot booking",
                           "Capacity-aware slot booking with overbooking guardrails per zone.",
                           ["node_js", "postgresql", "redis"], 95, "medium"),
                _component("ROUTE-002", "Route optimization",
                           "Heuristic vehicle routing for same-day grocery deliveries.",
                           ["python"], 120, "high"),
            ]),
    _budget("BUD-2024-008", "StyleLoop", "ecommerce", "US",
            "Fashion returns automation and resale", "python", 2024, [
                _component("RET-001", "Returns automation",
                           "Self-service returns with label generation and refund orchestration.",
                           ["python", "postgresql"], 80, "low"),
                _component("RESALE-002", "Resale listing pipeline",
                           "Automated grading and relisting of returned items into a resale catalog.",
                           ["python"], 70, "medium", ["RET-001"]),
            ]),
    _budget("BUD-2023-009", "MediTrack", "healthcare", "ES",
            "Patient appointment and EHR integration", "java", 2023, [
                _component("APPT-001", "Appointment scheduling",
                           "Slot-based scheduling with clinician calendars and reminder notifications.",
                           ["java", "postgresql"], 100, "medium"),
                _component("EHR-002", "EHR integration (HL7/FHIR)",
                           "FHIR-based integration for patient demographics and encounter sync.",
                           ["java"], 130, "high"),
            ]),
    _budget("BUD-2024-010", "CareLink", "healthcare", "US",
            "Telehealth video consultation platform", "node_js", 2024, [
                _component("VIDEO-001", "Video consultation",
                           "WebRTC video sessions with waiting room and clinician handoff.",
                           ["node_js", "redis"], 140, "high"),
                _component("RX-002", "e-Prescription module",
                           "Electronic prescription issuance with pharmacy routing.",
                           ["node_js", "postgresql"], 90, "medium"),
            ]),
    _budget("BUD-2024-011", "VitalSense", "healthcare", "DE",
            "Remote patient monitoring ingestion", "python", 2024, [
                _component("INGEST-001", "Device data ingestion",
                           "Streaming ingestion of vitals from BLE devices with backfill on reconnect.",
                           ["python", "redis"], 110, "high"),
                _component("ALERT-002", "Threshold alerting",
                           "Per-patient threshold alerting with escalation to care team.",
                           ["python", "postgresql"], 75, "medium", ["INGEST-001"]),
            ]),
    _budget("BUD-2023-012", "AssemblyOne", "industrial", "DE",
            "Factory IoT telemetry and OEE dashboards", "python", 2023, [
                _component("TELE-001", "Telemetry collector",
                           "MQTT telemetry collection from PLCs with edge buffering.",
                           ["python", "postgresql"], 120, "high"),
                _component("OEE-002", "OEE dashboard",
                           "Overall Equipment Effectiveness dashboards with shift breakdowns.",
                           ["react", "python"], 80, "medium", ["TELE-001"]),
            ]),
    _budget("BUD-2024-013", "LogiChain", "industrial", "ES",
            "Warehouse management and slotting", "java", 2024, [
                _component("WMS-001", "Warehouse management core",
                           "Inbound/outbound flows, putaway and pick-path generation.",
                           ["java", "postgresql"], 150, "high"),
                _component("SLOT-002", "Dynamic slotting",
                           "Velocity-based slotting recommendations recomputed nightly.",
                           ["python"], 85, "medium"),
            ]),
    _budget("BUD-2024-014", "GridWatch", "industrial", "US",
            "Energy grid anomaly detection", "python", 2024, [
                _component("ANOM-001", "Anomaly detection",
                           "Streaming anomaly detection on substation telemetry with seasonal baselines.",
                           ["python", "redis"], 135, "high"),
                _component("VIZ-002", "Operator console",
                           "Real-time operator console with acknowledgment workflow.",
                           ["react", "node_js"], 90, "medium"),
            ]),
    _budget("BUD-2023-015", "EduForge", "other", "ES",
            "LMS with adaptive learning paths", "ruby_on_rails", 2023, [
                _component("LMS-001", "Course delivery",
                           "Course content delivery with progress tracking and quizzes.",
                           ["ruby_on_rails", "postgresql"], 100, "medium"),
                _component("ADAPT-002", "Adaptive path engine",
                           "Adaptive sequencing of lessons based on assessment performance.",
                           ["python"], 95, "high", ["LMS-001"]),
            ]),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(BUDGETS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generados {len(BUDGETS)} presupuestos en {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
