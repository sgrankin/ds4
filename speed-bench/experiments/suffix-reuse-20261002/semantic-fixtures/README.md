# Fact-edit probes

The code arm changes only active_code from 2718 to 3141. The secret arm
changes only private_secret from 6829 to REDACTED. Both compare the edited
context against a suffix cached from the original register, and each asks
only about its changed field. The fresh baseline must itself answer correctly
before a reuse answer is treated as evidence of semantic recovery.

The oracle constructs one rendered user turn with a normal assistant prefix
and no-thinking marker. A contains the instructions plus neutral padding;
B contains neutral padding followed by the register; C contains neutral
padding followed by the final question and assistant prefix. Padding repeats
plain-text note tokens and truncates to the requested exact token length.
All parts are explicit token segments, so there is no later re-tokenization
that can move the A/B/C boundaries. Padding may end within a sentence.

At A=1024, B=256, C=512 or 1024, tail=1/64/128 keeps the complete prompt
fixed. Only the old suffix-cache endpoint moves backward. Larger fresh tails
therefore replace more stale C rows and recompute more of the final question;
the changed B is always recomputed through every layer. Greedy answers and
gold-versus-old complete answer log probabilities are separate measurements
from full-vocabulary KL and teacher-forced delta NLL.
