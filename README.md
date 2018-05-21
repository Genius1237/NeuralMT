# NeuralMT

Machine Translation for Sanskrit

The grammar of sanskrit is very unique, arguably even expressible in a Backus Naur Form(BNF)

Recent advances in MT are the use of a Seq2Seq model for translation, pioneered by `Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H., & Bengio, Y. (2014).`

This project aims to create such a model for Sanskrit and add in elements that incorporate sanskrit's grammar.

# Dataset used

A Sanskrit-English parallel corpus is needed. 2 different corpora were found
- One is a Sanskrit Translation of the Bible, available at http://sanskritbible.in/
- Second is an English Translation of the Valmiki Ramayana, available at https://www.valmiki.iitk.ac.in/

Both have been preprocessed to a list of sentences form and can be found in `Data`

# Folder Structure

The main code can be found in `Code/keras_end_to_end.py`

Different datasets used are in `Data/`

# Status
Work still in progress. Will be updated with any advancements
