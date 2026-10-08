from collections.abc import Iterator

import pytest
from openai import OpenAIError

from assistant.cli import repl


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class EchoAgent:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def ask(self, text: str) -> str:
        self.asked.append(text)
        if text == "boom":
            raise OpenAIError("API down")
        return f"echo: {text}"


async def run_session(lines: list[str]) -> tuple[EchoAgent, list[str]]:
    agent = EchoAgent()
    inputs: Iterator[str] = iter(lines)
    output: list[str] = []

    async def read_line(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    await repl(agent, read_line, output.append)  # type: ignore[arg-type]
    return agent, output


@pytest.mark.anyio
async def test_messages_go_to_the_agent_until_exit() -> None:
    agent, output = await run_session(["გამარჯობა", "  ", "გასვლა", "never sent"])
    assert agent.asked == ["გამარჯობა"]
    assert "echo: გამარჯობა" in output
    assert output[-1] == "ნახვამდის!"


@pytest.mark.anyio
async def test_help_is_answered_locally() -> None:
    agent, output = await run_session(["დახმარება"])
    assert agent.asked == []
    assert any("მაგალითები" in line for line in output)


@pytest.mark.anyio
async def test_errors_do_not_end_the_session() -> None:
    agent, output = await run_session(["boom", "კიდევ"])
    assert "შეცდომა: API down" in output
    assert agent.asked == ["boom", "კიდევ"]
