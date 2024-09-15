# Format of the Dataset

## Inference:

only two files are needed, named `test.tsv` and `test.txt`.

- In `test.tsv`, two columns, separated by a tab:
  1. Utterance unique ID
  2. Absolute path of the audio file

- In `test.txt`, five columns, separated by a tab: 
  1. uttid: Utterance unique ID
  2. origin_ds: Name of the database the audio belongs to
  3. speaker: Speaker ID 
  4. attacker: The name of the speech synthesis model used to generate the audio
  5. label: [spoof/bonafide]

NOTE: 
- In `test.txt`, columns 2, 3, and 4 are used for subsequent data analysis after inference is complete. If you only need the inference results, these three fields can be filled with any values. But better set origin_ds to the same, be
- But better to set origin_ds to the same value because the inference function will calculate the EER and acc for all samples belonging to the same `origin_ds`. If `origin_ds` has only spoof or bonafide samples, the EER for that `origin_ds` will `nan`. If each sample belongs to a different `origin_ds`, it will result in unnecessary computations.

