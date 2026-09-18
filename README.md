# Talkover

An inference gateway that wraps the Gander full-duplex interaction model as an OpenAI Realtime compatible service, able to run on a single Apple Silicon machine.

See [docs/DESIGN.md](docs/DESIGN.md) for the design document.

```bash
uv sync --extra dev --extra mps
uv run talkover check -c configs/serve.example.yaml
uv run talkover serve -c configs/serve.example.yaml
```
