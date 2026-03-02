# Changelog

All notable changes to **daily-stonks-clean** will be documented in this file.

This repository contains:
- `dailystonks/engine/` — report generation / “engine” logic and configuration
- `dailystonks-delivery/` — delivery service (API + queue + runner) and test suite

Format inspired by *Keep a Changelog*.  
This project follows pragmatic versioning (tags/releases will be added as the project stabilizes).

## [Unreleased]
### Added
- (reserved)

### Changed
- (reserved)

### Fixed
- (reserved)

---

## [0.1.0] - 2026-03-02
### Added
- Clean, minimal public repository layout containing only operational code:
  - engine package and config
  - delivery service package
- Fast unit test suite (default `pytest` run).
- Gated integration test suite that:
  - spins up a temporary Docker Postgres instance
  - blocks outbound SMTP during tests
  - writes a minimal required data fixture when needed
- Support banner injection helper and tests to guarantee:
  - insertion when missing
  - idempotence (safe to run multiple times)
- GitHub Actions workflow to run unit tests on push/PR, with integration available manually/nightly (as configured).

### Notes
- This release represents the “clean baseline” intended for CI and public collaboration.