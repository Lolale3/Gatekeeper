"""LIVE extractor demo -- requires Ollama running with qwen2.5:7b pulled.

Runs a few fixture invoices through the REAL extractor and compares each field to
the gold answer, showing the type-aware correctness verdict and whether the value
was grounded in the source. This is the first time the pipeline touches a model.

    PYTHONPATH=. python examples/extract_demo.py
"""
from gatekeeper import FixtureInvoiceLoader, invoice_schema, is_correct
from gatekeeper.extract import LLMExtractor, OllamaClient


def main() -> None:
    schema = invoice_schema()
    extractor = LLMExtractor(OllamaClient(model="qwen2.5:7b"))
    examples = FixtureInvoiceLoader().load()[:4]

    print("Extracting with a real local model (Ollama / qwen2.5:7b)...\n")
    for ex in examples:
        record = extractor.extract(ex.to_record(), schema)
        print(f"{ex.id}:")
        if record.meta.get("extraction_error"):
            print("  EXTRACTION ERROR:", record.meta["extraction_error"])
        for spec in schema.fields:
            f = record.fields[spec.name]
            gold = ex.ground_truth.get(spec.name)
            mark = "OK" if is_correct(f.value, gold, spec.dtype) else "XX"
            ground = "grounded" if f.source_span else "not-found"
            print(f"  [{mark}] {spec.name:15} got={str(f.value)!r:22} "
                  f"gold={str(gold)!r:22} ({ground})")
        print()


if __name__ == "__main__":
    main()