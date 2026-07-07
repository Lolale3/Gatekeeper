"""§1b (real): the parser recovers source_text and the mapped gold fields from the
dataset's nested string-encoded rows. Tests the parsing offline (no download)."""
import json

from gatekeeper.data.real_invoice import parse_row, _to_obj


def _make_row():
    # reconstruct the dataset's storage format: JSON strings whose values are
    # Python-repr strings (single quotes), as mychen76/...ocr_v1 stores them.
    ocr = ['Invoice no: 40378170', 'Date of issue:', '10/15/2012',
           'Seller:', 'Patel, Thompson and Montgomery', '356 Kyle Vista',
           'Client:', 'Jackson, Odonnell and Jackson', 'SUMMARY', 'Total', '66,00']
    labels = {'header': {'invoice_no': '40378170', 'invoice_date': '10/15/2012',
                         'seller': 'Patel, Thompson and Montgomery 356 Kyle Vista'},
              'summary': {'total_gross_worth': '$66,00'}}
    raw_data = json.dumps({"ocr_words": str(ocr)})
    parsed_data = json.dumps({"xml": "", "json": str(labels)})
    return raw_data, parsed_data


def test_parse_row_recovers_text_and_fields():
    raw_data, parsed_data = _make_row()
    source_text, gold = parse_row(raw_data, parsed_data)

    assert "Invoice no: 40378170" in source_text
    assert "Jackson, Odonnell and Jackson" in source_text          # OCR ordering preserved
    assert gold["invoice_number"] == "40378170"
    assert gold["invoice_date"] == "10/15/2012"
    assert gold["vendor_name"].startswith("Patel, Thompson and Montgomery")
    assert gold["total_amount"] == "$66,00"


def test_to_obj_handles_json_and_repr_and_passthrough():
    assert _to_obj('{"a": 1}') == {"a": 1}                          # json
    assert _to_obj("{'a': 1}") == {"a": 1}                          # python repr
    assert _to_obj({"a": 1}) == {"a": 1}                            # already an object
    assert _to_obj("plain text") == "plain text"                    # unparseable -> passthrough


def test_parse_row_tolerates_direct_dict_labels():
    # some rows may already be parsed into dicts by the datasets library
    raw = {"ocr_words": ["Invoice no: 5", "Total", "10.00"]}
    labels = {"header": {"invoice_no": "5", "invoice_date": "2026-01-01", "seller": "Acme"},
              "summary": {"total_gross_worth": "10.00"}}
    source_text, gold = parse_row(raw, labels)
    assert gold["invoice_number"] == "5"
    assert gold["vendor_name"] == "Acme"


def test_number_parsing_handles_european_decimals():
    from gatekeeper.data.matching import to_number
    assert to_number("66,00") == 66.0            # EU decimal comma
    assert to_number("1.234,56") == 1234.56      # EU thousands + decimal
    assert to_number("$1,240.00") == 1240.0      # US thousands + decimal
    assert to_number("1,240") == 1240.0          # US thousands, no decimal
    assert to_number("2340.75") == 2340.75


def test_vendor_gold_strips_address():
    from gatekeeper.data.real_invoice import _company_name
    assert _company_name("Patel, Thompson and Montgomery 356 Kyle Vista, New James, MA") \
        == "Patel, Thompson and Montgomery"
    assert _company_name("Acme Freight Co.") == "Acme Freight Co."
    assert _company_name(None) is None