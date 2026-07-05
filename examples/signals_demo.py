"""Signals demo (offline, deterministic via FakeClient). Reproduces the exact
extraction the live model produced on inv-001 -- the stray '#4471' and the
normalized '1240' -- and shows the three signals reacting:

  * rules            catch the leading '#' on the identifier   (grounding is blind to it)
  * grounding        is now happy with the normalized 1240     (§2 literal-find said not-found)
  * self_consistency reads how much the model agrees with itself

    PYTHONPATH=. python examples/signals_demo.py
"""
from gatekeeper import invoice_schema, Record
from gatekeeper.extract import LLMExtractor, FakeClient
from gatekeeper.signals import SelfConsistencySignal, GroundingSignal, RuleSignal

SOURCE = "INVOICE #4471  Date: 2026-03-14  Total: $1,240.00  From: Acme Freight Co."

# what the live model actually returned on inv-001 (note the '#' and normalized 1240)
PRIMARY = ('{"invoice_number":"#4471","invoice_date":"2026-03-14",'
           '"total_amount":"1240","vendor_name":"Acme Freight Co."}')

# five self-consistency samples: the model mostly agrees, but wavers once on the
# identifier (drops the '#') and once on the amount (misreads 1250)
SAMPLES = [
    '{"invoice_number":"#4471","invoice_date":"2026-03-14","total_amount":"1240","vendor_name":"Acme Freight Co."}',
    '{"invoice_number":"4471", "invoice_date":"2026-03-14","total_amount":"1240.00","vendor_name":"Acme Freight Co."}',
    '{"invoice_number":"#4471","invoice_date":"2026-03-14","total_amount":"1240","vendor_name":"Acme Freight Co."}',
    '{"invoice_number":"#4471","invoice_date":"2026-03-14","total_amount":"1250","vendor_name":"Acme Freight Co."}',
    '{"invoice_number":"#4471","invoice_date":"2026-03-14","total_amount":"1240","vendor_name":"Acme Freight Co."}',
]


def main() -> None:
    schema = invoice_schema()
    record = LLMExtractor(FakeClient(PRIMARY)).extract(
        Record(id="inv-001", source_text=SOURCE), schema)

    sampler = LLMExtractor(FakeClient(SAMPLES))
    for gen in (SelfConsistencySignal(sampler, n=len(SAMPLES)), GroundingSignal(), RuleSignal()):
        record = gen.generate(record, schema)

    print(f"source: {SOURCE}\n")
    print(f"{'field':16}{'value':14} signal vector")
    print("-" * 66)
    for spec in schema.fields:
        f = record.fields[spec.name]
        sigs = "  ".join(f"{s.name}={s.value:.2f}" for s in f.signals)
        print(f"{spec.name:16}{str(f.value)!r:14} {sigs}")
    print("\nnote: invoice_number rules=0.50 flags the stray '#'; grounding stays "
          "1.00 (blind to it).\n      total_amount grounding=1.00 -- the normalized "
          "1240 is now correctly grounded.")


if __name__ == "__main__":
    main()