# Blind user test protocol (10-15 friends)

**Goal:** do people rate the model's outfits above color-rule outfits and random ones?

## Prepare
1. Use one fixed wardrobe (yours, ~50 items). For each of the 4 occasions, take the top-3 outfits from the **model**, the top-3 from the **color-rule** scorer (`ColorRulePairScorer`), and 3 **random** valid outfits (same templates).
2. Render each outfit as a collage image (PIL) with a neutral file name. Keep a private key sheet: `image -> source`.
3. Create a Google Form: for each shown image, a 1-5 rating ("How good is this outfit for <occasion>?"). Shuffle the order per respondent (Google Forms: shuffle question order, or make 3 form variants).

## Run
* Tell friends the outfits are mixed and unlabeled. Ask for consent before using any photos of them; do not collect names.
* After the test, delete any personal data you collected.

## Analyse
* Mean rating per source with n (respondents x outfits) and, ideally, a 95% bootstrap CI or a paired comparison per respondent.
* Report it even if the model does not win clearly - say so in the README.
