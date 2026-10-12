"""Live Demo of Phase 1: `from raglib import parser` on a generated Legal PDF."""

from __future__ import annotations

import json
from pathlib import Path
import pymupdf as fitz

from raglib import parser


def build_demo_legal_pdf(output_pdf: Path) -> Path:
    doc = fitz.open()
    p1 = (
        "SUPREME COURT OF INDIA - REPORTABLE\n"
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPEAL NO. 4821 OF 2024\n\n"
        "Alpha Technologies Pvt. Ltd. v. Union of India\n"
        "Dated: 14th October, 2024\n\n"
        "PART I - PRELIMINARY AND DEFINITIONS\n"
        "CHAPTER I - GENERAL PROVISIONS\n"
        "Section 1. Short title and definitions.-\n"
        "(1) This Act may be called THE DIGITAL DATA PROTECTION AND ARBITRATION ACT, 2024.\n"
        "(2) In this Act, unless the context otherwise requires:\n"
        '(a) "Confidential Information" means any proprietary data, trade secret, or juris-\n'
        "dictional record disclosed by a Data Fiduciary;\n"
        '(b) "Tribunal" shall mean the Appellate Tribunal established under Section 18 of the Telecom Regulatory Authority of India Act, 1997.\n'
        "Page 1 of 2\n"
    )
    p2 = (
        "SUPREME COURT OF INDIA - REPORTABLE\n"
        "Section 2. Obligations of Data Fiduciary.-\n"
        "(1) Every Data Fiduciary shall implement reasonable security safeguards to prevent personal data breach.\n"
        "(a) The Data Fiduciary shall not transfer sensitive personal data outside the territory without prior approval.\n"
        "Provided that nothing in this sub-section shall restrict cross-border transfers notified under Article 21 of the Constitution of India.\n"
        "Explanation 1.- For the purposes of this section, as held in Justice K.S. Puttaswamy v. Union of India, (2017) 10 SCC 1 and AIR 1973 SC 1461, privacy is an intrinsic right.\n"
        "Notwithstanding anything contained in Section 14 of the Arbitration and Conciliation Act, 1996, any dispute shall be resolved within ninety days.\n"
        "Page 2 of 2\n"
    )
    for txt in (p1, p2):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 555, 800), txt, fontsize=10)
    doc.save(str(output_pdf))
    doc.close()
    return output_pdf


if __name__ == "__main__":
    sample_pdf = build_demo_legal_pdf(Path("sample_legal_judgment.pdf"))

    # One-line command using the requested import style: `from raglib import parser`
    parsed_doc = parser.parse(sample_pdf)

    print("=== 1. PARSED SUMMARY ===")
    print(json.dumps(parsed_doc.summary(), indent=2))

    print("\n=== 2. LEGAL HIERARCHY TREE (EXTRACTED VIA REGEX) ===")
    for node in parsed_doc.flat_sections:
        print(f"  [{node.node_type}] {node.full_path} -> {node.title or node.text[:65]}")

    print("\n=== 3. LEGAL CITATIONS (EXTRACTED VIA REGEX) ===")
    for cit in parsed_doc.citations:
        print(f"  [{cit.citation_type}] {cit.normalized}")

    print("\n=== 4. LEGAL DEFINITIONS (EXTRACTED VIA REGEX) ===")
    for defn in parsed_doc.definitions:
        print(f"  '{defn.term}' => {defn.definition}")

    print("\n=== 5. LEGAL OBLIGATIONS / PROVISOS / NON-OBSTANTE CLAUSES ===")
    for ob in parsed_doc.obligations:
        print(f"  [{ob.category}] ({ob.section_path}) -> {ob.statement}")

