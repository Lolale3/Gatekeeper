"""Preview the synthetic freight emails (offline, always works). Shows the messy
broker prose the LLM must parse -- values buried in human language, varied phrasing,
informal formats, ambiguity -- next to the ground truth we planted. No regex gets
these; the model has to understand the text.

To see the LLM actually extract and the confidence layer gate them (needs Ollama):
    python -m gatekeeper.app --freight --html freight.html

    python examples/freight_demo.py
"""
from gatekeeper.data.freight import SyntheticFreightLoader, freight_schema


def main() -> None:
    schema = freight_schema()
    print(f"freight schema fields: {', '.join(schema.field_names())}\n")
    for ex in SyntheticFreightLoader(n=4).load():
        print(f"--- {ex.id} (broker email) ---")
        print(ex.source_text)
        print("ground truth:", ex.ground_truth.values)
        print()


if __name__ == "__main__":
    main()