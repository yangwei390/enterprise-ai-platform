from pydantic import BaseModel


class CalculatorArgs(BaseModel):
    expression: str


class EchoArgs(BaseModel):
    text: str


class CurrentTimeArgs(BaseModel):
    timezone: str | None = None
