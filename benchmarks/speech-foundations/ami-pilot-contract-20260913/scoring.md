# AMI speaker-word scoring contract

`scripts/score_speaker_words.py` compares the independent timed AMI word reference to a canonical schema-2 `speaker-transcript` artifact.

It reports:

- ordinary WER over the complete normalized word sequence;
- concatenated permutation-invariant speaker-attributed WER (`cp_sawer`), using an exact minimum-cost assignment between reference and system speaker streams;
- substitutions, deletions and insertions for ordinary WER;
- assigned stream errors plus explicit unlabelled-word errors for cpSA-WER.

Normalization is NFKC + case folding, Unicode letters/numbers, and internal apostrophes (`nfkc-casefold-unicode-alnum-internal-apostrophe-v1`). System speaker IDs are arbitrary and are never assumed to equal AMI participant IDs. A word assigned to the wrong speaker is represented by a deletion from the correct stream and insertion into the wrong stream; an unlabelled system word is excluded from speaker streams and charged separately. This makes speaker attribution errors visible even when lexical WER is zero.

The scorer rejects duplicate/unknown JSON keys, non-finite or unsorted timing, malformed speaker accounting, excessive input, and output overwrite. Six focused scorer tests and the combined 21-test corpus-contract suite pass. An idealized 179-word AMI pilot self-check produces WER 0 and cpSA-WER 0 under the optimal four-speaker permutation.

Absolute pilot metrics always retain `qualified:false`. Production qualification still requires a ratified corpus budget, a pinned external reference-system result, and more than one selected excerpt.
