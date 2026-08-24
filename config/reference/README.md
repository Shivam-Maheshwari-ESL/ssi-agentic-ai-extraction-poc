# Reference data ("golden set")

**Empty by default, and deliberately so.** No values are seeded from the sample documents:
a BIC directory built from two samples would make the pipeline look accurate on those files
and wrong everywhere else.

## How absence is handled

When a file here is missing, a lookup returns `available=False`, the confidence blend uses a
**neutral** score (so an unknown BIC neither raises nor lowers confidence), and the
document-level drift check reports `INDETERMINATE` rather than firing on every document. ISO
3166 country and ISO 4217 currency membership is always available through `pycountry`, so those
kinds are checked whether or not anything is placed here.

Populating these files switches the signal on with no code change. Loader:
`src/ssi_extractor/validators/reference_data.py`.

## Expected files

### `bic_directory.csv`
Header row required. Only `bic` is used; the other columns are for human reference.
An 11-character BIC also matches on its 8-character institution prefix.

```csv
bic,institution,country
DAKVDEFF,Clearstream Banking AG,DE
CEDELULL,Clearstream Banking S.A.,LU
```

### `country_pset_map.csv`
The depository a market's instructions are expected to settle at. Used by the cross-field rule
engine and as a confidence signal, never as a hard constraint.

```csv
country,pset_bic
AT,DAKVDEFF
FR,SICVFRPPXXX
```

Country may be an alpha-2 code, an alpha-3 code or an ISO name; it is canonicalised to alpha-2
on load.

### `account_patterns.yaml`
Regexes matched against the whole normalised account identifier (separators removed,
upper-cased). One list per market.

```yaml
FR: ['\d{11}']
GB: ['\d{8}']
LU: ['CBL\d{5}', '\d{5}']
```

## Provenance discipline

Any set added here should be traceable to an authoritative feed (SWIFT BIC directory, ANNA,
the institution's own rule book). Mark provisional data as such in a comment and treat drift
detection as advisory until a real feed is in place, because a wrong reference set produces
confident wrong answers — worse than no reference set at all.
