# Model changelog

Every change to `model/params.json` is recorded here by the weekly check (or by hand), with the reason.

- 2026-10-04 · v1 · Initial live rule. Longs: 3 most liquid extreme movers, buy at the close, sell at +0.25×ATR or after 5 sessions, no stop. Shorts: off (best short rule earned +0.13%/trade on the holdout with single losses up to −98%; every protective stop turned it negative).
