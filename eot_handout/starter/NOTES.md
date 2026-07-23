# Notes

**What signal the model uses:**
Our model uses a calibrated Logistic Regression over 15 causal, manually engineered prosodic features extracted strictly from the audio preceding each pause. Key discriminative signals include final-200ms energy decay, pause index characteristics, mid-turn filler detection via energy variance trailing tails, pre-boundary speech lengthening ratios (normalized by typical syllable durations), and F0 relative variance. The model is lightweight, explainable, and adheres strictly to the non-anticipatory causality requirement by updating turn-level accumulator contexts only *after* feature extraction.

**Where it still fails:**
The model's primary failure mode remains first-pause End-Of-Turns (`pause_index=0`), as contextual features (like turn running-average segment durations) have no prior history to normalize against. Additionally, it can be confused by sharp mid-turn cutoffs or mid-turn conversational filler words that do not exhibit standard stable energy patterns. 

**What I would do with one more day:**
With more time, I would incorporate speaker-specific normalization (e.g., dynamic z-scoring of F0 and energy per speaker/turn) to better handle diverse voices. I would also explore sequential learning using a lightweight causal LSTM or GRU to model the trajectory of prosodic features across pauses, rather than treating each pause as an independent static snapshot, and invest time in cross-lingual tuning to equalize English and Hindi threshold biases.
