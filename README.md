# Fill-in-the-blank NN's

Originally forked from PyTorch example code. All the changes are in the word_language_model folder.

These models are intended to simulate the alternatives that humans generate in the [Rational Speech Acts model](https://nlp.stanford.edu/pubs/monroe2015learning.pdf). They're essentially 'fill-in-the-blank' language models.

## Project motivation

Language is rife with ‘implicatures’: meanings that aren’t made explicit. An example might be, “Jim met with *some* of the candidates.” Here, the speaker doesn’t explicitly state that Jim met with some but not all of the candidates, but the listener would assume this – because if the speaker meant that Jim met with all of the candidates, the speaker would have said, “Jim met with *all* of the candidates.” A crucial part of the reasoning by the listener is the understanding that ‘all’ is an alternative to ‘some’. If it wasn’t clear that the speaker might instead have said ‘all,’ then the listener would have needed to reason over the ambiguous meanings of ‘some.’ This example raises the question of how the listener comes up with ‘all’ as an alternative, and in general, the question of how these alternatives are generated, which is the crux of this work. We have two main questions. First, how do humans generate alternatives? Second, how well are these alternatives predicted by computational models? The code in this repository is aimed at the latter.
