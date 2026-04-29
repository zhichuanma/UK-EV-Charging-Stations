## Column notes (as of Stage 7)

- `huff_score_<scene>` : **LEGACY, FROZEN**. Computed by the pre-Stage-7 POI
  labeling pipeline with a 500 m hard cutoff. Preserved as historical artifact;
  no longer updated. Not used by any active code in the Stage 1+ architecture.
- `station_attractiveness` : dimensionless, `log(1 + TotalCapacity_kW)`. Added
  in Stage 7. Used by Stage 1.4 Layer-2 station sampler.
- `label` : **LEGACY**. Early station-scene labels based on surrounding POI
  structure. Not used in the new matching architecture but preserved for
  backward compatibility.
