"""Entry point for `adk web agents` and `adk run agents/contractlens_agent`."""

from contractlens.pipeline import ContractLens

root_agent = ContractLens.from_settings().agent
