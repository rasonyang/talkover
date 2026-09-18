# Test fixtures

## `zh_order_16k.wav`

A 5.2 s Mandarin clip, mono pcm16 at 16 kHz (~168 KB), used by `tests/engine/test_asr.py`
to check that an ASR backend returns Chinese text with timestamps.

Spoken sentence (the only Chinese string in the test suite, it is runtime data):

```
您好，我的订单号是八六四二，请帮我查一下物流。
```

Generated on macOS with the bundled `Tingting` (zh_CN) voice — `say -v '?'` lists the
available Chinese voices — and converted to the wav format the engine uses:

```bash
say -v Tingting -o /tmp/tt.aiff "您好，我的订单号是八六四二，请帮我查一下物流。"
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/tt.aiff tests/fixtures/zh_order_16k.wav
```

Tests match the transcription tolerantly (a few expected characters, not the whole
sentence), because Whisper punctuation and digit spelling vary between models.
