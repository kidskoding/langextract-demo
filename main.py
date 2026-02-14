"""
LangExtract Demo — Standalone Script
=====================================
This script demonstrates the core langextract workflow:
  1. Define a prompt describing what to extract
  2. Provide few-shot examples teaching the LLM your extraction schema
  3. Run extraction on new text
  4. Save results to JSONL
  5. Generate an interactive HTML visualization

Usage:
  python main.py

Requires:
  - GEMINI_API_KEY set in environment (or .env file)
  - pip install langextract python-dotenv
"""

import langextract as lx
import textwrap

# ---------------------------------------------------------------------------
# Step 1: Define the extraction prompt
# ---------------------------------------------------------------------------
# This tells the LLM *what kinds of things* to extract.
# Be specific about categories and mention "exact text" to avoid paraphrasing.
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# ---------------------------------------------------------------------------
# Step 2: Provide few-shot examples
# ---------------------------------------------------------------------------
# Examples teach the LLM the output format. Each ExampleData has:
#   - text: the example input
#   - extractions: what should be found in that text
#
# Each Extraction has:
#   - extraction_class: the category (e.g. "character", "emotion")
#   - extraction_text: the EXACT substring from the text
#   - attributes: extra metadata about this extraction
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]

# ---------------------------------------------------------------------------
# Step 3: Run extraction on new text
# ---------------------------------------------------------------------------
# lx.extract() returns an AnnotatedDocument with:
#   - .text: the original input
#   - .extractions: list of Extraction objects with char_interval positions
#   - .document_id: auto-generated unique ID
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# ---------------------------------------------------------------------------
# Step 4: Save to JSONL
# ---------------------------------------------------------------------------
# JSONL = one JSON object per line, easy to stream and append to.
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# ---------------------------------------------------------------------------
# Step 5: Generate interactive HTML visualization
# ---------------------------------------------------------------------------
# Creates color-coded, interactive HTML with play/pause controls.
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)
    else:
        f.write(html_content)
