# Applications

- `api/`: FastAPI service exposing backend capabilities and an
  OpenAI-compatible local generation interface.
- `web/`: lightweight React/TypeScript demo and benchmark viewer.

Applications consume `src/slm_lab/`; they do not own model-conversion or
benchmark logic.
