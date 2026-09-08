# Requirement formalization tranche

`assess_requirements` now exposes `compatibility_status` alongside the established eligibility output. The expanded vocabulary is `EXACT_FIT`, `DIRECTLY_ANSWERABLE`, `PARTIAL_FIT`, `INDIRECT_SUPPORT`, `REQUIRES_ADDITIONAL_DATA`, `INCOMPATIBLE`, and `UNKNOWN`. The status is derived only from field-level observed capabilities, mapping types, and explicitly supplied direct perturbational evidence classifications.

Explicit mismatch yields `INCOMPATIBLE`; missing comparator/design/metadata remains `UNKNOWN`; derived, ortholog, broader, narrower, related, or unresolved mappings remain `INDIRECT_SUPPORT`; no requested requirements yields `REQUIRES_ADDITIONAL_DATA`. This does not alter the frozen alpha eligibility interface or promote indirect/negative evidence into direct fit.
