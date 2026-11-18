# Existing models and transferable lessons

## Existing NFL ATS model
Known design includes opponent-adjusted offensive/defensive EPA per play, success rate, recency weighting, prior-season regression, fitted margin relationship and home-field advantage. Existing workflow uses nflverse/nflseedR and immutable prediction history. Preserve model version, source commit, prediction time, feature cutoff and target definition on import.

CBS pool picks are not executable-market recommendations. Pool pick grading and market betting must remain separate. Earlier workflow used Pinnacle around 20 minutes pre-kickoff as an official market reference, with retry and labelled median fallback. This repository must preregister its own close definition and preserve `MARKET_MEDIAN_FALLBACK`/`FEW_BOOKS` flags.

## Existing player-prop research lessons (props excluded here)
- Participation-conditioned distributions must not fall back to unconditional samples without flagging.
- Feature normalization and implied-point baselines must be training/as-of only.
- Avoid arbitrary simulation draw caps that truncate tails.
- Future-data perturbation tests should prove historical predictions do not change.
These are lessons from targeted review, not a claim of a complete audit or verified fixes.

## Tennis copy-trading lesson
Prior copying analysis showed negative copied returns under tested delay assumptions. Signal quality is not execution quality; measure fees, delays, slippage and fills directly.
