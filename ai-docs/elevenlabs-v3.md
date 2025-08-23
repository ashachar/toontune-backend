## ElevenLabs v3 (August 2025): Markdown Setup & Usage Guide

### What’s New in ElevenLabs v3?
- **New expression controls** and support for *unlimited speakers* in dialogs.[1]
- Easily specify `eleven_v3` as the model ID for Text to Speech API calls.[2][1]
- Dialog mode, extra emotion/audio tags, 70+ languages, advanced endpoints for music & sound effects, and global TTS API with low-latency regions.[3][2][1]
- Expanded API endpoints: music generation, sound effects, voice management, and more.[4][2]

***

### Basic Usage

#### 1. Installation (Python SDK)
```bash
pip install elevenlabs
```
- For TypeScript/JS: Use the `@elevenlabs` SDK (see official docs).

#### 2. API Authentication
You need an API key from your ElevenLabs account. Add it to your environment:
```bash
export ELEVENLABS_API_KEY="your-api-key-here"
```
Or pass as a parameter in SDK config.

#### 3. Basic Text-to-Speech Example (Python)
```python
from elevenlabs.client import ElevenLabs
client = ElevenLabs()
audio = client.text_to_speech.convert(
    text="Hello! This is ElevenLabs v3 speaking.",
    voice_id="YOUR_VOICE_ID",
    model_id="eleven_v3",  # Specify v3 model
    output_format="mp3_44100_128",
)
with open("output.mp3", "wb") as f:
    f.write(audio)
```
- Adjust `voice_id`, `text`, and `output_format` as needed.[5][2]

#### 4. Using Dialog Mode & Multiple Speakers
- Use the Dialog/Text-to-Dialogue API endpoint.
- Pass a list of utterances with speaker tags.
- Example (JSON format; actual API details in the official docs):

```json
{
  "model_id": "eleven_v3",
  "dialog": [
    {"speaker": "Alice", "text": "Hi, Bob!"},
    {"speaker": "Bob", "text": "Hello, Alice!"}
  ]
}
```
- Result: Output audio switches voices based on speaker tags.[2][1]

#### 5. Advanced Controls
- Use audio tags to control *emotion*, *style*, and *timing* (see official prompting guide).
- Example with emotion:
  ```
  <emotion="cheerful">This is a happy voice.</emotion>
  ```

#### 6. Generate Music/Sound Effects
- Use new endpoints for music or SFX:
    - **Compose music:** Text prompt → original music.
    - **Text-to-Sound Effects:** Text prompt → realistic SFX.[3][4][2]

```json
{
  "prompt": "Create an energetic background for an action scene."
}
```
- Endpoints and payload details: refer to ElevenLabs API changelog/docs.[4][2]

***

### Helpful Markdown Shortcuts

#### Markdown to Voice Guide
If you create markdown content and want to convert it to audio:
- Use open-source projects like [`markdown-to-elevenlabs`](https://github.com/danmenzies/markdown-to-elevenlabs).[6]
- Place your text in a `.md` file, run the script, and it’ll generate an audio file using your ElevenLabs API key.

***

### Other Features
- **Speech to Speech:** Convert your own audio into another voice with emotional preservation.[3]
- **Conversational AI:** Real-time agents with low-latency interaction.[7][2][3]
- **Audio Native:** Embed audio players on your site/blog.[3]

***

### Official Resources
- [ElevenLabs Changelog - v3 details](https://elevenlabs.io/docs/changelog/2025/8/20)[2]
- [API Docs: Full endpoint and SDK documentation](https://elevenlabs.io/docs/)[5][7][2]
- [Prompts & advanced usage guide](https://elevenlabs.io/docs/prompts/)[2]

***

### Summary Table

| Feature                      | How to Use                                       | Markdown Tip                  |
|----------------------------- |--------------------------------------------------|-------------------------------|
| Text-to-Speech v3            | Set `model_id="eleven_v3"`                       | Create `.md` for your script  |
| Multiple Speakers/Dialog     | Use dialog endpoint w/ speaker tags               | Structure dialogue in markdown|
| Music/SFX Generation         | Use music/SFX endpoint with prompt                | Summarize mood in prompt      |
| Speech-to-Speech             | Upload audio, select target voice                 | Attach audio snippets         |
| Voice Management/Cloning     | Use voice management endpoints                    | Document your workflow        |
| SDK Integration              | Official Python/JS SDKs, version 2.9+             | Reference code blocks in docs |

***

**Tip:** For maximum expression and features, always refer to the current [official docs & changelog]. The v3 model is now the recommended option for expressive TTS and creative audio generation.[1][5][2]

[1](https://the-decoder.com/elevenlabs-releases-its-v3-model-with-new-expression-controls-and-support-for-unlimited-speakers/)
[2](https://elevenlabs.io/docs/changelog/2025/8/20)
[3](https://www.fahimai.com/elevenlabs)
[4](https://tech.az/en/posts/elevenlabs-has-introduced-a-new-feature-that-creates-music-for-videos-5503)
[5](https://pypi.org/project/elevenlabs/)
[6](https://github.com/danmenzies/markdown-to-elevenlabs)
[7](https://blog.getbind.co/2025/08/19/how-to-use-elevenlabs-voice-ai-in-your-applications/)
[8](https://elevenlabs.io/docs/changelog/2025/8/11)
[9](https://elevenlabs.io/docs/changelog/2025/8/4)
[10](https://elevenlabs.io/docs/changelog)
[11](https://apidog.com/blog/how-to-use-elevenlabs-mcp-server/)