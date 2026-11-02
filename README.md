# YuE2 Music Cover — Auto Vocal PRO v2

ComfyUI **Music Cover (YuE2)** workflow plus a custom node that promotes the `Ins` melody from SheetSage2 to `Vocal`, analyzes whether the new lyrics cover the extracted melody, and can align section labels or **optionally** repeat *only* lyrics the user wrote. No external Python dependencies.

**Stable names — no renames:** workflow `YuE2_Cover_AutoVocal_PRO_V2.json`, subgraph `Music Cover (YuE2) — Auto Vocal PRO v2`, custom node class `YuE2AutoVocalPRO`, directory `custom_nodes/yue2_auto_vocal_pro/`.

## Structure

```text
YuE2_Cover_AutoVocal_PRO_V2.json
README.md
.gitignore
custom_nodes/
  yue2_auto_vocal_pro/
    __init__.py           # ComfyUI registration and input/output signature
    engine.py             # ABC Ins → Vocal routing; native Z4| rests
    alignment.py          # section matching and lyric coverage diagnostics
tests/
    test_engine.py
    test_alignment.py
    test_workflow.py
```

The distributable ZIP **does not contain model weights, MP3/audio, extracted soundtrack ABC, generated songs, API keys or user lyrics**. Replace the reference filename in `LoadAudio` with your own audio after importing.

## Install (Windows / ComfyUI)

1. Stop ComfyUI fully.
2. Copy the *contents* of `custom_nodes/yue2_auto_vocal_pro/` into the pre-existing `ComfyUI/custom_nodes/yue2_auto_vocal_pro/` directory, overwriting the existing Python files. **Do not rename the directory.**
3. Restart ComfyUI, then load **`YuE2_Cover_AutoVocal_PRO_V2.json`**.
4. In `LoadAudio`, load a reference audio that you have rights to use. Enter **Style** in the subgraph widget and your **Lyrics** in the separate multiline node (both ship empty). The workflow retains a 190-second maximum duration, `melody` mode and `all_of_rest` section policy.
5. The external `Seed — randomize / fixed` node defaults to **randomize**. Use **fixed** and a numeric seed when comparing musical results.
6. Inside the subgraph, examine `PreviewAny` **44** (source ABC), **49** (transformed ABC), and **48** (conversion + lyric-coverage report).

**Important:** Style and Lyrics are intentionally blank for version control. With `strict = true` (retained from the working baseline), executing an empty Lyrics field is expected to fail with a clear validation error. Enter lyrics before generating.

You must have the ComfyUI YuE2 native nodes and its SheetSage2/audio encoder available, and the corresponding model checkpoint files installed. Model file names and all prior node/link IDs are preserved from the user's corrected baseline.

## What's new: lyric alignment (inside the existing custom node)

Only **one new internal widget** has been added to `YuE2AutoVocalPRO`, without adding ports or nodes to the outer workflow:

| `lyrics_alignment` | Effect | Changes the text you wrote? |
|---|---|---|
| `match_sections` **(default)** | Estimates section coverage and relabels ABC `%` vocal sections to match Lyrics `[Section]` **only when their counts match** | **No** |
| `report_only` | Outputs the original routed ABC and original Lyrics, appending diagnostics only | **No** |
| `repeat_existing` | Same section alignment; additionally repeats the *exact existing lines* to target about 2 bars per lyric line (maximum 8 repetitions), **only when the number of vocal and lyric sections matches** | **Yes — explicitly opt-in** |

The `repeat_existing` option does **not** create or paraphrase words, and never changes or generates musical notes. It may be inappropriate for narrative lyrics; evaluate the text before using it. Automatic syllable-to-note synchronization is **not implemented**; YuE2 itself determines phrasing. All numerical coverage values (estimated syllables, notes per syllable, bars per line) are **heuristics**, not a phonetic aligner or a guarantee against vocalizing "la la la".

For instance, when the routed score has `% verse` followed by a later `% verse` but the user has `[Verse]` and `[Chorus]`, `match_sections` changes the **second score comment** to `% chorus` if those are the only two vocal sections. Intro, interlude, solo and outro remain instrumental with `all_of_rest`. If there are more/fewer lyrical sections than vocal sections, the node warns instead of guessing the mapping.

## Existing options (preserved)

- `section_policy`: `all_of_rest` (default, excludes instrumental-intent sections), `verses_and_choruses`, `all_sections`.
- `instrumental_handling`: `replace_with_rests` (default, native multimeasure `Z4|`), `keep_original`.
- `existing_vocal`: `preserve` or `overwrite`. The saved outer workflow currently uses `overwrite` as in the user baseline.
- `instrumental_tags` / `vocal_tags`: optional comma/newline-separated overrides **for classifying the ABC sections**, not text instructions to YuE2.
- `strict`: boolean; raises on empty lyrics or a complete lack of eligible converted/already-vocal content.
- `planning_mode`: native YuE2 `melody` (default) / `full` switch; other YuE2, sampler and VAE controls are unchanged.

### How to write Lyrics

Use section headers with actual sung words underneath, e.g.:

```text
[Verse]
First sung line
Second sung line

[Chorus]
First chorus line
Second chorus line
```

These are **illustrative only**; the JSON has no sample lyrics. Avoid using an eight-line lyric for ~60 bars of singing without checking coverage. Matching the original lyric phrasing and syllable counts typically helps covers. `[Guitar Solo]` and other custom tags can remain as written, but YuE2 does **not** promise Suno-like interpretation of arbitrary tags. `repeat_existing` repeats lines as text only; it does not create new ABC phrases or split existing measures.

## Architecture and unchanged connections

```text
LoadAudio ─> SheetSage2AudioToABC ─> PreviewAny(44)
                                               └─> YuE2AutoVocalPRO(47)
                                                      ├─> PreviewAny(49): ABC
                                                      ├─> PreviewAny(48): report
                                                      ├─> YuE2GenerateMusic.abc
                                                      └─> YuE2GenerateMusic.lyrics
Lyrics node(50) ─> subgraph ─> YuE2AutoVocalPRO.lyrics
SeedNode(51) ──> subgraph ─> SeedNode(34) ─> YuE2GenerateMusic + KSampler
YuE2GenerateMusic ─> KSampler ─> VAEDecodeAudio ─> SaveAudioAdvanced
```

The outer subgraph input count, all link IDs, all node IDs and the subgraph identifier are preserved. The only modification inside the node layout is appending the `lyrics_alignment` widget to the existing node 47. Existing exports/ports `abc`, `lyrics`, `report` are unchanged.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use **synthetic musical fixtures**, not copyrighted reference melodies. They cover note routing, multibar rests, chord retention, instrumentation sections, lyrics preservation and opt-in repeating, warnings, JSON input/links and unchanged interface identities. The test suite checks the **Python code and workflow structure**, but is not an end-to-end acoustic test on ComfyUI or a quality guarantee.

If the generated audio still contains non-lexical vocalizations, inspect PreviewAny(48) for `bars_per_line`, `estimated_syllables`, and `notes_per_estimated_syllable`, then try a lyric with more complete lines matching the melodic phrases, or explicitly select `repeat_existing` for a controlled experiment. Run with a fixed seed to compare conditions. Note that `max_duration` is a ceiling; the song may end early and long prompts/ABC can reduce the effective token budget.

## References

- [ComfyUI YuE2 Music Generation Guide](https://docs.comfy.org/tutorials/audio/yue2/yue2)
- [YuE ABC editing reference](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/abc-editing.md)

## Version-control guidance

No Git remote, token or personal Git account is included. Create a repository in your own Git hosting service, commit these files, and keep locally installed checkpoints and source audio outside Git. `.gitignore` excludes common generated Python and ComfyUI artifacts.
