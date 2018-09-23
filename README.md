ex1 and ex2 both contain a main.py file that runs without parameters.
The main.py produces .png image files of the graphs that show an indication
regarding the nature and the position of the attack in the recording.

ex1 graphs shows three unusual bandwidth peeks that are probably the attacks.
ex2 shows unusual consecutive large 16-offset packets jumps in the low 4 bytes
of the payload of the packets.

In the "deep" folder I added a training code for a deep language model (LM) that
can be effective for more unknown "zero-day" attacks. This is a raw solution
(aka with bad coding style) that takes a long time to train, so I don't know
whether or not it converges correctly.
Run deep\main_preprocess.py without parameters before running deep\main_train.py