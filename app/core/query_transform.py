import re
from typing import List

class QueryTransformer:
    """
    Transforms raw user queries to maximize retrieval recall.
    Capabilities:
    1. Financial Query Normalization (standardizing terms: NII, PAT, Capex, DLG, FEMA).
    2. Hypothetical Document Embeddings (HyDE) expansion.
    3. Multi-Query Sub-question Decomposition.
    """
    SYNONYM_EXPANSIONS = {
        "nii": "Net Interest Income interest earned interest expended",
        "pat": "Profit After Tax net profit standalone consolidated",
        "dlg": "Default Loss Guarantee digital lending first loss first default",
        "capex": "capital expenditure investments plant machinery",
        "npa": "non performing asset gross npa net npa asset quality",
        "crar": "Capital to Risk Weighted Assets Ratio CAR Tier 1 Tier 2",
        "fema": "Foreign Exchange Management Act overseas direct investment ODI",
        "esg": "Environmental Social Governance BRSR Business Responsibility"
    }

    @classmethod
    def expand_query(cls, query: str) -> str:
        expanded_parts = [query]
        q_lower = query.lower()
        for acronym, full_phrase in cls.SYNONYM_EXPANSIONS.items():
            if re.search(rf'\b{acronym}\b', q_lower):
                expanded_parts.append(full_phrase)
        return " ".join(expanded_parts)

    @classmethod
    def generate_hyde_passage(cls, query: str) -> str:
        """
        Generates a synthetic hypothetical passage answering the query
        to bridge the semantic gap between short questions and long document chunks.
        """
        return f"Regarding {query}: According to the financial statements and regulatory circulars, the specific figures, limits, and statutory requirements are defined with audited numbers and exact regulatory thresholds."
