#!/usr/bin/env python3
"""Run one independently configured Qualcomm AI Hub compile stage."""

from slm_lab.deployment.qualcomm.ai_hub import stage_main


if __name__ == "__main__":
    raise SystemExit(stage_main("compile"))
